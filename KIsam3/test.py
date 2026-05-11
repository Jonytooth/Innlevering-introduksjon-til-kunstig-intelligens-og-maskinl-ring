"""
test.py — Run on Thunder Compute after training.
Evaluates untrained vs fine-tuned SAM 3 on the 10 test images.
Saves comparison.png — download it before stopping your instance.
"""

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from huggingface_hub import login
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ── Config (must match preprocess.py and train.py) ────────────────────────────
IMG_DIR      = "images"
MASK_DIR     = "masks"
EMBED_DIR    = "embeddings"
CKPT_DIR     = "checkpoints"
HF_TOKEN     = "--change--"
TEXT_PROMPT  = "camouflaged person"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 1024

print(f"Using device: {DEVICE}")


# ── Load test split ───────────────────────────────────────────────────────────
def load_test_split():
    te_imgs  = list(np.load(os.path.join(CKPT_DIR, "test_imgs.npy"),  allow_pickle=True))
    te_masks = list(np.load(os.path.join(CKPT_DIR, "test_masks.npy"), allow_pickle=True))
    print(f"Loaded {len(te_imgs)} test images")
    return te_imgs, te_masks


# ── SAM 3 decoder wrapper (must match train.py) ───────────────────────────────
# Updated SAM 3 decoder wrapper
class SAM3Decoder(nn.Module):
    def __init__(self, model, processor):
        super().__init__()
        self.model     = model
        self.processor = processor

    def forward(self, backbone_out):
        # The processor requires original image dimensions to scale results
        state = {
            "backbone_out": {
                k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in backbone_out.items()
            },
            "original_height": IMG_SIZE,
            "original_width": IMG_SIZE,
            "image_height": IMG_SIZE,
            "image_width": IMG_SIZE,
        }
        
        with torch.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu", dtype=torch.bfloat16):
            text_out = self.model.backbone.forward_text(
                [TEXT_PROMPT], device=DEVICE
            )
            state["backbone_out"].update(text_out)
            state["geometric_prompt"] = self.model._get_dummy_prompt()
            
            # Now state contains the required height/width keys
            output = self.processor._forward_grounding(state)

        masks_logits = output.get("masks_logits", None)
        if masks_logits is None or masks_logits.shape[0] == 0:
            return None, None

        scores = output.get("scores", None)
        best   = scores.argmax().item() if scores is not None else 0
        logits = masks_logits[best].unsqueeze(0).float()
        score  = scores[best].item() if scores is not None else 0.0
        return logits, score


# ── Metrics ───────────────────────────────────────────────────────────────────
downsample = nn.AdaptiveAvgPool2d((256, 256))

def compute_metrics(logits, gt):
    pred  = (torch.sigmoid(logits) > 0.5).float()
    gt    = downsample(gt.unsqueeze(0)).to(DEVICE)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    dice  = (2 * inter / (union + 1e-6)).item()
    iou   = (inter / (union - inter + 1e-6)).item()
    return dice, iou, pred.squeeze().cpu().numpy()


# ── Evaluate ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(decoder, te_imgs, te_masks, label):
    decoder.eval()
    all_dice, all_iou, results = [], [], []

    for img_path, mask_path in tqdm(zip(te_imgs, te_masks),
                                     total=len(te_imgs),
                                     desc=f"Evaluating [{label}]"):
        fname      = os.path.splitext(os.path.basename(img_path))[0]
        embed_path = os.path.join(EMBED_DIR, f"{fname}.pt")
        backbone   = torch.load(embed_path, weights_only=False)

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        gt   = torch.from_numpy(
            (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        ).unsqueeze(0)

        logits, score = decoder(backbone)

        if logits is None:
            all_dice.append(0.0)
            all_iou.append(0.0)
            if len(results) < 10:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((256, 256), Image.BILINEAR)
                gt_small = downsample(gt.unsqueeze(0)).squeeze().numpy()
                blank = np.zeros((256, 256), dtype=np.float32)
                results.append((
                    np.array(img), gt_small, blank,
                    os.path.basename(img_path), 0.0, 0.0
                ))
            continue

        if logits.shape[-2:] != (256, 256):
            logits = nn.functional.interpolate(
                logits, size=(256, 256), mode="bilinear", align_corners=False
            )

        dice, iou, pred = compute_metrics(logits, gt)
        all_dice.append(dice)
        all_iou.append(iou)

        if len(results) < 10:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((256, 256), Image.BILINEAR)
            gt_small = downsample(gt.unsqueeze(0)).squeeze().numpy()
            results.append((
                np.array(img), gt_small, pred,
                os.path.basename(img_path), dice, iou
            ))

    mean_dice = np.mean(all_dice)
    mean_iou  = np.mean(all_iou)
    print(f"[{label}]  Dice: {mean_dice:.4f}  |  IoU: {mean_iou:.4f}")
    return mean_dice, mean_iou, results


# ── Per-image table ───────────────────────────────────────────────────────────
def print_per_image(results_un, results_tr):
    print("\n─── Per-image results ───")
    print(f"{'Image':<40} {'Untrained Dice':>14} {'Untrained IoU':>13} {'Trained Dice':>12} {'Trained IoU':>11}")
    print("-" * 95)
    for (_, _, _, fname, d_un, i_un), (_, _, _, _, d_tr, i_tr) in zip(results_un, results_tr):
        print(f"{fname:<40} {d_un:>14.4f} {i_un:>13.4f} {d_tr:>12.4f} {i_tr:>11.4f}")


# ── Visual comparison ─────────────────────────────────────────────────────────
def visualise(results_un, results_tr):
    n   = min(10, len(results_un))
    fig = plt.figure(figsize=(16, n * 4))
    gs  = gridspec.GridSpec(n, 4, figure=fig, hspace=0.4, wspace=0.3)

    for row in range(n):
        img, gt, pred_un, fname, d_un, i_un = results_un[row]
        _,   _,  pred_tr, _,     d_tr, i_tr = results_tr[row]

        for col, (data, title) in enumerate([
            (img,     f"Image\n{fname}"),
            (gt,      "Ground Truth"),
            (pred_un, f"Untrained SAM 3\nDice={d_un:.3f} IoU={i_un:.3f}"),
            (pred_tr, f"Fine-tuned SAM 3\nDice={d_tr:.3f} IoU={i_tr:.3f}"),
        ]):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(data, cmap=None if col == 0 else "gray")
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    plt.suptitle(
        f'SAM 3 — Untrained vs Fine-tuned | Prompt: "{TEXT_PROMPT}"',
        fontsize=12, y=1.01
    )
    plt.savefig("comparison.png", bbox_inches="tight", dpi=150)
    print("\ncomparison.png saved — download it before stopping your instance!")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    login(token=HF_TOKEN)

    print("\nLoading SAM 3...")
    model     = build_sam3_image_model().to(DEVICE)
    processor = Sam3Processor(model)
    model.eval()

    te_imgs, te_masks = load_test_split()

    # 1. Evaluate UNTRAINED
    print("\n─── Untrained model ───")
    untrained = SAM3Decoder(model, processor).to(DEVICE)
    dice_un, iou_un, results_un = evaluate(untrained, te_imgs, te_masks, "Untrained")

    # 2. Load best checkpoint and evaluate TRAINED
    print("\n─── Fine-tuned model ───")
    best_path = os.path.join(CKPT_DIR, "best.pth")
    assert os.path.exists(best_path), "No best.pth found — run train.py first!"
    trained = SAM3Decoder(model, processor).to(DEVICE)
    ckpt    = torch.load(best_path, map_location=DEVICE)
    trained.load_state_dict(ckpt["model_state"])
    dice_tr, iou_tr, results_tr = evaluate(trained, te_imgs, te_masks, "Fine-tuned")

    # 3. Per-image breakdown
    print_per_image(results_un, results_tr)

    # 4. Summary
    print("\n══════════════════════════════════════")
    print("           RESULTS SUMMARY            ")
    print("══════════════════════════════════════")
    print(f"  Untrained SAM 3  │ Dice: {dice_un:.4f}  IoU: {iou_un:.4f}")
    print(f"  Fine-tuned SAM 3 │ Dice: {dice_tr:.4f}  IoU: {iou_tr:.4f}")
    print(f"  Improvement      │ Dice: {dice_tr-dice_un:+.4f}  IoU: {iou_tr-iou_un:+.4f}")
    print("══════════════════════════════════════")

    # 5. Visual comparison
    visualise(results_un, results_tr)
