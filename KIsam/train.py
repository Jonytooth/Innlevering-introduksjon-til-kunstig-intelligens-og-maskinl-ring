import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from segment_anything import sam_model_registry
from tqdm import tqdm
import monai
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Config ────────────────────────────────────────────────────────────────────
IMG_DIR        = "images"
MASK_DIR       = "masks"
EMBED_DIR      = "embeddings"           # pre-computed embeddings saved here
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
MODEL_TYPE     = "vit_h"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 2
NUM_EPOCHS     = 20
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
RANDOM_SEED    = 42
IMG_SIZE       = 1024
OUTPUT_DIR     = "checkpoints"
N_IMAGES       = 1000                 
TEST_SIZE      = 0.2                   

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EMBED_DIR,  exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"Using device: {DEVICE}")


# ── Data split ────────────────────────────────────────────────────────────────
def build_splits():
    # 1. Gather all available image and mask files
    img_files  = sorted([f for f in os.listdir(IMG_DIR)  if f.lower().endswith(".jpg")])
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.lower().endswith(".png")])

    # 2. Ensure the dataset contains enough images
    assert len(img_files) >= N_IMAGES, f"Found only {len(img_files)} images, need {N_IMAGES}"

    # 3. Create a list of indices and shuffle them to ensure random selection
    indices = list(range(len(img_files)))
    random.seed(RANDOM_SEED) # Ensures reproducibility of the random selection
    random.shuffle(indices)

    # 4. Select the first 50 random indices from the shuffled list
    selected_indices = indices[:N_IMAGES]
    
    img_files  = [img_files[i] for i in selected_indices]
    mask_files = [mask_files[i] for i in selected_indices]

    img_paths  = [os.path.join(IMG_DIR,  f) for f in img_files]
    mask_paths = [os.path.join(MASK_DIR, f) for f in mask_files]

    # 5. Split the 50 randomly selected images into training and test sets
    tr_imgs, te_imgs, tr_masks, te_masks = train_test_split(
        img_paths, mask_paths,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    
    print(f"Total (randomly sampled): {N_IMAGES} | Train: {len(tr_imgs)} | Test: {len(te_imgs)}")
    return tr_imgs, te_imgs, tr_masks, te_masks


# ── Step 1: Pre-compute and save all embeddings ───────────────────────────────
def precompute_embeddings(all_img_paths, sam):
    """
    Runs the frozen image encoder once per image and saves the result to disk.
    This step is skipped for images that already have a saved embedding.
    """
    sam.image_encoder.eval()
    print("\n─── Pre-computing image embeddings (runs once, then cached) ───")

    for img_path in tqdm(all_img_paths, desc="Encoding images"):
        fname      = os.path.splitext(os.path.basename(img_path))[0]
        embed_path = os.path.join(EMBED_DIR, f"{fname}.pt")

        if os.path.exists(embed_path):
            continue  # already computed, skip

        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img = torch.from_numpy(
            np.array(img, dtype=np.float32)        # keep in [0,255] for SAM
        ).permute(2, 0, 1).unsqueeze(0).to(DEVICE) # 1 C H W

        with torch.no_grad():
            embedding = sam.image_encoder(img)      # 1 C H W  (small feature map)

        torch.save(embedding.cpu(), embed_path)

    print(f"All embeddings saved to '{EMBED_DIR}/'")


# ── Dataset (uses saved embeddings, not raw images) ───────────────────────────
class EmbeddingDataset(Dataset):
    """
    Loads pre-computed image embeddings instead of raw images.
    The encoder is never called during training — much faster.
    """
    def __init__(self, img_paths, mask_paths):
        self.img_paths  = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load embedding from disk
        fname      = os.path.splitext(os.path.basename(self.img_paths[idx]))[0]
        embed_path = os.path.join(EMBED_DIR, f"{fname}.pt")
        embedding  = torch.load(embed_path, weights_only=True).squeeze(0)  # C H W

        # Load mask
        mask = Image.open(self.mask_paths[idx]).convert("L")
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        mask = torch.from_numpy(
            (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        ).unsqueeze(0)                              # 1 H W

        return embedding, mask, self.img_paths[idx]


# Dataset for evaluation (needs raw image for visualisation)
class EvalDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        self.img_paths  = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        fname      = os.path.splitext(os.path.basename(self.img_paths[idx]))[0]
        embed_path = os.path.join(EMBED_DIR, f"{fname}.pt")
        embedding  = torch.load(embed_path, weights_only=True).squeeze(0)

        # Raw image for visualisation
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        mask = Image.open(self.mask_paths[idx]).convert("L")
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        mask = torch.from_numpy(
            (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        ).unsqueeze(0)

        return embedding, img, mask, self.img_paths[idx]


# ── Loss ──────────────────────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(sigmoid=True, reduction="mean")
        self.bce  = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.bce(logits, targets)


# ── SAM decoder-only wrapper ──────────────────────────────────────────────────
class SAMDecoder(nn.Module):
    """
    Only runs the prompt encoder + mask decoder.
    Image encoder is never called during training.
    """
    def __init__(self, sam):
        super().__init__()
        self.sam = sam

    def forward(self, embeddings):
        """embeddings: (B, C, H, W) pre-computed image features."""
        B = embeddings.shape[0]
        sparse_emb, dense_emb = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        masks_list = []
        for i in range(B):
            low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings=embeddings[i].unsqueeze(0).to(DEVICE),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            masks_list.append(low_res_masks)
        return torch.cat(masks_list, dim=0)          # B 1 256 256


def build_model():
    sam   = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    model = SAMDecoder(sam).to(DEVICE)
    for p in model.sam.image_encoder.parameters():
        p.requires_grad = False
    for p in model.sam.mask_decoder.parameters():
        p.requires_grad = True
    for p in model.sam.prompt_encoder.parameters():
        p.requires_grad = True
    return model, sam


# ── Metrics ───────────────────────────────────────────────────────────────────
downsample = nn.AdaptiveAvgPool2d((256, 256))

def compute_metrics(logits, masks):
    preds = (torch.sigmoid(logits) > 0.5).float()
    masks = downsample(masks.to(DEVICE))
    inter = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice  = (2 * inter / (union + 1e-6)).cpu().numpy()
    iou   = (inter / (union - inter + 1e-6)).cpu().numpy()
    return dice, iou, preds


# ── Evaluate untrained (needs full SAM, no embeddings yet) ────────────────────
@torch.no_grad()
def evaluate_untrained(sam, te_imgs, te_masks):
    """Run untrained SAM directly (before embeddings exist)."""
    print("\n─── Evaluating untrained model ───")
    sam.eval()
    all_dice, all_iou, results = [], [], []

    for img_path, mask_path in tqdm(zip(te_imgs, te_masks), total=len(te_imgs)):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32)
        img_t  = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        mask_t = torch.from_numpy(
            (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        ).unsqueeze(0).unsqueeze(0)

        embedding = sam.image_encoder(img_t)
        sparse_emb, dense_emb = sam.prompt_encoder(points=None, boxes=None, masks=None)
        logits, _ = sam.mask_decoder(
            image_embeddings=embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )

        dice, iou, preds = compute_metrics(logits, mask_t)
        all_dice.extend(dice.tolist())
        all_iou.extend(iou.tolist())

        if len(results) < 4:
            results.append((
                torch.from_numpy(img_np / 255.0).permute(2, 0, 1),
                downsample(mask_t).squeeze().numpy(),
                preds.squeeze().cpu().numpy(),
                img_path,
            ))

    mean_dice = np.mean(all_dice)
    mean_iou  = np.mean(all_iou)
    print(f"[Untrained]  Dice: {mean_dice:.4f}  |  IoU: {mean_iou:.4f}")
    return mean_dice, mean_iou, results


# ── Evaluate trained ──────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_trained(model, te_dl, label="Fine-tuned"):
    print(f"\n─── Evaluating {label} model ───")
    model.eval()
    all_dice, all_iou, results = [], [], []

    for embeddings, imgs, masks, paths in tqdm(te_dl):
        embeddings = embeddings.to(DEVICE)
        logits     = model(embeddings)
        dice, iou, preds = compute_metrics(logits, masks)
        all_dice.extend(dice.tolist())
        all_iou.extend(iou.tolist())

        if not results:
            for i in range(len(imgs)):
                results.append((
                    imgs[i].cpu(),
                    downsample(masks[i].unsqueeze(0)).squeeze().numpy(),
                    preds[i].squeeze().cpu().numpy(),
                    paths[i],
                ))

    mean_dice = np.mean(all_dice)
    mean_iou  = np.mean(all_iou)
    print(f"[{label}]  Dice: {mean_dice:.4f}  |  IoU: {mean_iou:.4f}")
    return mean_dice, mean_iou, results


# ── Visualise ─────────────────────────────────────────────────────────────────
def visualise_comparison(results_un, results_tr, n=4):
    n   = min(n, len(results_un))
    fig = plt.figure(figsize=(16, n * 4))
    gs  = gridspec.GridSpec(n, 4, figure=fig, hspace=0.4, wspace=0.3)

    for row in range(n):
        img_t, gt, pred_un, path = results_un[row]
        _,     _,  pred_tr, _   = results_tr[row]
        img_np = img_t.permute(1, 2, 0).numpy()
        fname  = os.path.basename(path)

        for col, (data, title) in enumerate([
            (img_np,  f"Image\n{fname}"),
            (gt,      "Ground Truth"),
            (pred_un, "Untrained SAM"),
            (pred_tr, "Fine-tuned SAM"),
        ]):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(data, cmap=None if col == 0 else "gray")
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    plt.suptitle("Untrained vs Fine-tuned SAM", fontsize=13, y=1.01)
    plt.savefig("comparison.png", bbox_inches="tight", dpi=150)
    print("Comparison saved → comparison.png")
    plt.show()


# ── Training ──────────────────────────────────────────────────────────────────
def train(model, tr_dl):
    trainable = (
        list(model.sam.mask_decoder.parameters()) +
        list(model.sam.prompt_encoder.parameters())
    )
    optimiser = AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS)
    criterion = CombinedLoss().to(DEVICE)
    best_loss = float("inf")

    print("\n─── Training (decoder only — encoder skipped each epoch) ───")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for embeddings, masks, _ in tqdm(tr_dl, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            embeddings = embeddings.to(DEVICE)
            masks      = downsample(masks.to(DEVICE))
            optimiser.zero_grad()
            logits = model(embeddings)
            loss   = criterion(logits, masks)
            loss.backward()
            optimiser.step()
            ep_loss += loss.item()

        ep_loss /= len(tr_dl)
        scheduler.step()
        print(f"  Epoch {epoch:02d} | loss={ep_loss:.4f}")

        ckpt = {"epoch": epoch, "model_state": model.state_dict(), "loss": ep_loss}
        torch.save(ckpt, os.path.join(OUTPUT_DIR, "last.pth"))
        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save(ckpt, os.path.join(OUTPUT_DIR, "best.pth"))
            print(f"    ✓ New best saved (loss={best_loss:.4f})")

    return model


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Split data
    tr_imgs, te_imgs, tr_masks, te_masks = build_splits()
    all_imgs = tr_imgs + te_imgs

    # 2. Load SAM once (used for embedding pre-computation + untrained eval)
    print("\nLoading SAM...")
    model, sam = build_model()

    # 3. Evaluate UNTRAINED model BEFORE computing embeddings
    dice_un, iou_un, results_un = evaluate_untrained(sam, te_imgs, te_masks)

    # 4. Pre-compute embeddings for ALL images (skips already-computed ones)
    precompute_embeddings(all_imgs, sam)

    # 5. Build dataloaders using saved embeddings
    tr_ds  = EmbeddingDataset(tr_imgs, tr_masks)
    te_ds  = EvalDataset(te_imgs, te_masks)
    tr_dl  = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    te_dl  = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 6. Train (encoder never runs again after this point)
    model = train(model, tr_dl)

    # 7. Load best and evaluate
    best_ckpt = torch.load(os.path.join(OUTPUT_DIR, "best.pth"), map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state"])
    dice_tr, iou_tr, results_tr = evaluate_trained(model, te_dl)

    # 8. Summary
    print("\n══════════════════════════════════════")
    print("           RESULTS SUMMARY            ")
    print("══════════════════════════════════════")
    print(f"  Untrained SAM │ Dice: {dice_un:.4f}  IoU: {iou_un:.4f}")
    print(f"  Fine-tuned    │ Dice: {dice_tr:.4f}  IoU: {iou_tr:.4f}")
    print(f"  Improvement   │ Dice: {dice_tr - dice_un:+.4f}  IoU: {iou_tr - iou_un:+.4f}")
    print("══════════════════════════════════════")

    # 9. Visual comparison
    visualise_comparison(results_un, results_tr, n=4)
