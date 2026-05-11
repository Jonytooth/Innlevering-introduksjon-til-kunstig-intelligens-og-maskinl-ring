"""
preprocess.py — Run this on your LOCAL PC before going to Thunder Compute.
It picks 50 random images, splits them 80/20, runs the SAM 3 backbone
once per image and saves the embeddings to disk.
Upload the 'embeddings/' folder and 'checkpoints/' folder to Thunder Compute.
"""

import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from huggingface_hub import login

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ── Config ────────────────────────────────────────────────────────────────────
IMG_DIR      = "images"
MASK_DIR     = "masks"
EMBED_DIR    = "embeddings"
CKPT_DIR     = "checkpoints"
HF_TOKEN     = "--change--"
IMG_SIZE     = 1024
N_IMAGES     = 300
TEST_SIZE    = 0.2
RANDOM_SEED  = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(EMBED_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,  exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"Using device: {DEVICE}")
print("Note: preprocessing on CPU is slow but free. Let it run overnight if needed.")


# ── Pick 50 random images ─────────────────────────────────────────────────────
def build_splits():
    img_files  = sorted([f for f in os.listdir(IMG_DIR)  if f.lower().endswith(".jpg")])
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.lower().endswith(".png")])

    assert len(img_files)  >= N_IMAGES, f"Need {N_IMAGES} images, found {len(img_files)}"
    assert len(mask_files) >= N_IMAGES, f"Need {N_IMAGES} masks,  found {len(mask_files)}"

    # Random shuffle with fixed seed — reproducible
    combined = list(zip(img_files, mask_files))
    random.shuffle(combined)
    img_files, mask_files = zip(*combined)

    img_paths  = [os.path.join(IMG_DIR,  f) for f in img_files[:N_IMAGES]]
    mask_paths = [os.path.join(MASK_DIR, f) for f in mask_files[:N_IMAGES]]

    tr_imgs, te_imgs, tr_masks, te_masks = train_test_split(
        img_paths, mask_paths,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    # Save split so train.py and test.py use the exact same images
    np.save(os.path.join(CKPT_DIR, "test_imgs.npy"),  te_imgs)
    np.save(os.path.join(CKPT_DIR, "test_masks.npy"), te_masks)
    np.save(os.path.join(CKPT_DIR, "train_imgs.npy"), tr_imgs)
    np.save(os.path.join(CKPT_DIR, "train_masks.npy"),tr_masks)

    print(f"Total: {N_IMAGES} | Train: {len(tr_imgs)} | Test: {len(te_imgs)}")
    print(f"Split saved to '{CKPT_DIR}/'")
    return tr_imgs + te_imgs   # all 50 need embeddings


# ── Pre-compute embeddings ────────────────────────────────────────────────────
def precompute(all_img_paths, processor):
    print("\n─── Pre-computing embeddings (runs once, skips cached) ───")
    for img_path in tqdm(all_img_paths, desc="Encoding"):
        fname      = os.path.splitext(os.path.basename(img_path))[0]
        embed_path = os.path.join(EMBED_DIR, f"{fname}.pt")
        if os.path.exists(embed_path):
            continue

        pil_img = Image.open(img_path).convert("RGB")
        pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

        with torch.no_grad():
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16) if DEVICE == "cuda" else torch.no_grad():
                state = processor.set_image(pil_img)

        embedding = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in state["backbone_out"].items()
            if not callable(v)
        }
        torch.save(embedding, embed_path)

    print(f"Done! {len(os.listdir(EMBED_DIR))} embeddings saved to '{EMBED_DIR}/'")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Authenticate
    login(token=HF_TOKEN)

    # 2. Load SAM 3 (backbone only needed here)
    print("\nLoading SAM 3...")
    model     = build_sam3_image_model().to(DEVICE)
    processor = Sam3Processor(model)
    model.eval()

    # 3. Split and get all image paths
    all_imgs = build_splits()

    # 4. Pre-compute embeddings
    precompute(all_imgs, processor)

    print("\n✓ Preprocessing complete!")
    print("Now upload these two folders to Thunder Compute:")
    print(f"  → {EMBED_DIR}/   ({len(os.listdir(EMBED_DIR))} .pt files)")
    print(f"  → {CKPT_DIR}/    (split .npy files)")
    print("Also upload your images/ and masks/ folders for test.py visualisation.")
