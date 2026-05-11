import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from segment_anything import sam_model_registry
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Config ────────────────────────────────────────────────────────────────────
IMG_DIR        = "images"
MASK_DIR       = "masks"
EMBED_DIR      = "embeddings"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
MODEL_TYPE     = "vit_h"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 2
RANDOM_SEED    = 42
IMG_SIZE       = 1024
OUTPUT_DIR     = "checkpoints"
N_IMAGES       = 1000
TEST_SIZE      = 0.2

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

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

# ── Dataset ───────────────────────────────────────────────────────────────────
class EvalDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        fname = os.path.splitext(os.path.basename(self.img_paths[idx]))[0]
        embed_path = os.path.join(EMBED_DIR, f"{fname}.pt")
        embedding = torch.load(embed_path, weights_only=True).squeeze(0)

        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        mask = Image.open(self.mask_paths[idx]).convert("L")
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        mask = torch.from_numpy(
            (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        ).unsqueeze(0)

        return embedding, img, mask, self.img_paths[idx]

# ── Model ─────────────────────────────────────────────────────────────────────
class SAMDecoder(nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.sam = sam

    def forward(self, embeddings):
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

        return torch.cat(masks_list, dim=0)


def build_model():
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    model = SAMDecoder(sam).to(DEVICE)
    return model, sam

# ── Metrics ───────────────────────────────────────────────────────────────────
downsample = nn.AdaptiveAvgPool2d((256, 256))

def compute_metrics(logits, masks):
    preds = (torch.sigmoid(logits) > 0.5).float()
    masks = downsample(masks.to(DEVICE))

    inter = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

    dice = (2 * inter / (union + 1e-6)).cpu().numpy()
    iou = (inter / (union - inter + 1e-6)).cpu().numpy()

    return dice, iou, preds

# ── Untrained evaluation ─────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_untrained(sam, te_imgs, te_masks):
    print("─── Evaluating untrained model ───")
    sam.eval()

    all_dice, all_iou, results = [], [], []

    for img_path, mask_path in tqdm(zip(te_imgs, te_masks), total=len(te_imgs)):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        mask_t = torch.from_numpy(
            (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        ).unsqueeze(0).unsqueeze(0)

        # Use cached embedding instead of recomputing encoder
        fname = os.path.splitext(os.path.basename(img_path))[0]
        embed_path = os.path.join(EMBED_DIR, f"{fname}.pt")
        embedding = torch.load(embed_path, weights_only=True).to(DEVICE)
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

        if len(results) < 10:
            results.append((
                torch.from_numpy(img_np / 255.0).permute(2, 0, 1),
                downsample(mask_t).squeeze().numpy(),
                preds.squeeze().cpu().numpy(),
                img_path,
            ))

    return np.mean(all_dice), np.mean(all_iou), results

# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_trained(model, te_dl, label="Fine-tuned"):
    print(f"\n─── Evaluating {label} model ───")
    model.eval()

    all_dice, all_iou, results = [], [], []

    for embeddings, imgs, masks, paths in tqdm(te_dl):
        embeddings = embeddings.to(DEVICE)
        logits = model(embeddings)

        dice, iou, preds = compute_metrics(logits, masks)
        all_dice.extend(dice.tolist())
        all_iou.extend(iou.tolist())

        if len(results) < 10:
            for i in range(len(imgs)):
                results.append((
                    imgs[i].cpu(),
                    downsample(masks[i].unsqueeze(0)).squeeze().numpy(),
                    preds[i].squeeze().cpu().numpy(),
                    paths[i],
                ))

    mean_dice = np.mean(all_dice)
    mean_iou = np.mean(all_iou)

    print(f"[{label}] Dice: {mean_dice:.4f} | IoU: {mean_iou:.4f}")
    return mean_dice, mean_iou, results

# ── Visualise ─────────────────────────────────────────────────────────────────
def visualise_comparison(results_un, results_tr, n=10):
    n = min(n, len(results_un))
    fig = plt.figure(figsize=(16, n * 4))
    gs = gridspec.GridSpec(n, 4, figure=fig, hspace=0.4, wspace=0.3)

    for row in range(n):
        img_t, gt, pred_un, path = results_un[row]
        _, _, pred_tr, _ = results_tr[row]
        img_np = img_t.permute(1, 2, 0).numpy()
        fname = os.path.basename(path)

        for col, (data, title) in enumerate([
            (img_np, f"Image\n{fname}"),
            (gt, "Ground Truth"),
            (pred_un, "Reference"),
            (pred_tr, "Fine-tuned SAM"),
        ]):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(data, cmap=None if col == 0 else "gray")
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    plt.savefig("comparison.png", bbox_inches="tight", dpi=150)
    plt.show()

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tr_imgs, te_imgs, tr_masks, te_masks = build_splits()

    # Build separate models for true comparison
    untrained_model, untrained_sam = build_model()
    trained_model, _ = build_model()

    ckpt = torch.load(os.path.join(OUTPUT_DIR, "best.pth"), map_location=DEVICE)
    trained_model.load_state_dict(ckpt["model_state"])

    te_ds = EvalDataset(te_imgs, te_masks)
    te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    dice_un, iou_un, results_un = evaluate_untrained(untrained_sam, te_imgs, te_masks)
    dice_tr, iou_tr, results_tr = evaluate_trained(trained_model, te_dl)
    n = min(len(results_un), len(results_tr))
    visualise_comparison(results_un, results_tr, n=n)
