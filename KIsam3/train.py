"""
train.py — Run this on Thunder Compute (A100).
Requires embeddings/ and checkpoints/ folders uploaded from your PC.
The image encoder never runs here — training is fast and cheap.
"""

import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from huggingface_hub import login
from tqdm import tqdm
import monai

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_DIR    = "embeddings"
MASK_DIR     = "masks"
CKPT_DIR     = "checkpoints"
HF_TOKEN     = "--change--"
TEXT_PROMPT  = "camouflaged person"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS   = 10
LR           = 1e-5
WEIGHT_DECAY = 1e-4
RANDOM_SEED  = 42
IMG_SIZE     = 1024

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"Using device: {DEVICE}")


# ── Load split saved by preprocess.py ─────────────────────────────────────────
def load_splits():
    tr_imgs  = [p.replace("\\", "/") for p in np.load(os.path.join(CKPT_DIR, "train_imgs.npy"),  allow_pickle=True)]
    tr_masks = [p.replace("\\", "/") for p in np.load(os.path.join(CKPT_DIR, "train_masks.npy"), allow_pickle=True)]
    print(f"Train: {len(tr_imgs)} | Test split saved separately for test.py")
    return tr_imgs, tr_masks


# ── Dataset ───────────────────────────────────────────────────────────────────
class EmbeddingDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        self.img_paths  = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path   = self.img_paths[idx].replace("\\", "/")
        mask_path  = self.mask_paths[idx].replace("\\", "/")
        fname      = os.path.splitext(os.path.basename(img_path))[0]
        embed_path = os.path.join(EMBED_DIR, f"{fname}.pt")
        embedding  = torch.load(embed_path, weights_only=False)

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((256, 256), Image.NEAREST)
        mask = torch.from_numpy(
            (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        ).unsqueeze(0)

        return embedding, mask


# ── Loss ──────────────────────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(sigmoid=True, reduction="mean")
        self.bce  = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.bce(logits, targets)


# ── SAM 3 decoder wrapper ─────────────────────────────────────────────────────
class SAM3Decoder(nn.Module):
    """
    Only runs the transformer + segmentation head.
    Image encoder never called — embeddings are pre-computed.
    """
    def __init__(self, model, processor):
        super().__init__()
        self.model     = model
        self.processor = processor

    def forward(self, backbone_out):
        state = {
            "backbone_out": {
                k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in backbone_out.items()
            },
            "original_height": 1024,
            "original_width":  1024,
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Encode text prompt
            text_out = self.model.backbone.forward_text(
                [TEXT_PROMPT], device=DEVICE
            )
            state["backbone_out"].update(text_out)
            state["geometric_prompt"] = self.model._get_dummy_prompt()

            # Run transformer + segmentation head only
            output = self.processor._forward_grounding(state)

        masks_logits = output.get("masks_logits", None)
        if masks_logits is None or masks_logits.shape[0] == 0:
            return None

        scores = output.get("scores", None)
        best   = scores.argmax().item() if scores is not None else 0
        return masks_logits[best].unsqueeze(0).float()   # 1 1 H W


# ── Training ──────────────────────────────────────────────────────────────────
def train(decoder, tr_imgs, tr_masks):
    # Freeze backbone — only train transformer and segmentation head
    for p in decoder.model.backbone.parameters():
        p.requires_grad = False
    for p in decoder.model.transformer.parameters():
        p.requires_grad = True
    for p in decoder.model.segmentation_head.parameters():
        p.requires_grad = True

    trainable = (
        list(decoder.model.transformer.parameters()) +
        list(decoder.model.segmentation_head.parameters())
    )
    optimiser = AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS)
    criterion = CombinedLoss().to(DEVICE)
    downsample = nn.AdaptiveAvgPool2d((256, 256))
    best_loss  = float("inf")

    ds = EmbeddingDataset(tr_imgs, tr_masks)
    dl = DataLoader(ds, batch_size=None, shuffle=True, num_workers=0)

    print("\n─── Training SAM 3 (decoder only) ───")
    for epoch in range(1, NUM_EPOCHS + 1):
        decoder.train()
        ep_loss = 0.0
        n_valid = 0

        for backbone_out, mask in tqdm(dl, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            mask = mask.to(DEVICE)
            optimiser.zero_grad()

            logits = decoder(backbone_out)
            if logits is None:
                torch.cuda.empty_cache()
                continue

            gt = downsample(mask.unsqueeze(0))
            if logits.shape[-2:] != gt.shape[-2:]:
                logits = nn.functional.interpolate(
                    logits, size=gt.shape[-2:], mode="bilinear", align_corners=False
                )

            loss = criterion(logits, gt)
            loss.backward()
            optimiser.step()
            ep_loss += loss.item()
            n_valid += 1
            torch.cuda.empty_cache()

        if n_valid == 0:
            print(f"  Epoch {epoch:02d} | No detections — skipping")
            continue

        ep_loss /= n_valid
        scheduler.step()
        print(f"  Epoch {epoch:02d} | loss={ep_loss:.4f} ({n_valid}/{len(tr_imgs)} detections)")

        ckpt = {"epoch": epoch, "model_state": decoder.state_dict(), "loss": ep_loss}
        torch.save(ckpt, os.path.join(CKPT_DIR, "last.pth"))
        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save(ckpt, os.path.join(CKPT_DIR, "best.pth"))
            print(f"    ✓ New best saved (loss={best_loss:.4f})")

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print("Download checkpoints/best.pth before stopping your instance!")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    login(token=HF_TOKEN)

    print("\nLoading SAM 3...")
    model     = build_sam3_image_model().to(DEVICE)
    processor = Sam3Processor(model)
    model.eval()

    tr_imgs, tr_masks = load_splits()

    decoder = SAM3Decoder(model, processor).to(DEVICE)
    train(decoder, tr_imgs, tr_masks)
