import os
import random
import shutil
import cv2
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------
# CONFIGURATION - CHANGE THESE PATHS
# ---------------------------------------------------------
IMG_DIR = r"images"      # Folder containing original JPG images
GT_DIR  = r"masks"       # Folder containing ground truth PNG masks
OUT_DIR = r"output"   # Folder where selected samples and results are saved

# Set seed for reproducibility
random.seed(42)

# Create output directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# STAGE 1: SELECTION AND COPYING
# ---------------------------------------------------------

images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")]
groups = defaultdict(list)

for img in images:
    base = os.path.splitext(img)[0]
    parts = base.split("_")
    camo_type = parts[0].replace("dataset", "")
    groups[camo_type].append(base)

selected_samples = []

print("--- Step 1: Selecting and Copying Samples ---")

for camo_type in sorted(groups.keys(), key=lambda x: int(x)):
    items = groups[camo_type]
    choice = random.choice(items)
    
    img_src = os.path.join(IMG_DIR, choice + ".jpg")
    mask_src = os.path.join(GT_DIR, choice + ".png")

    if not os.path.exists(mask_src):
        continue

    img_dst = os.path.join(OUT_DIR, choice + ".jpg")
    mask_dst = os.path.join(OUT_DIR, choice + ".png")

    shutil.copy2(img_src, img_dst)
    shutil.copy2(mask_src, mask_dst)
    
    selected_samples.append(choice + ".jpg")
    print(f"Selected Type {camo_type}: {choice}.jpg")

# ---------------------------------------------------------
# STAGE 2: EDA & VALUE TRACKING
# ---------------------------------------------------------

print("\n--- Step 2: Running Analysis ---\n")

# Lists to store values for min/max calculation
all_r, all_g, all_b = [], [], []
all_brightness = []
all_contrast = []

for img_name in selected_samples:
    path = os.path.join(OUT_DIR, img_name)
    img = cv2.imread(path)

    if img is None:
        continue

    # RGB Intensity
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean_r, mean_g, mean_b = img_rgb.mean(axis=(0, 1))
    
    all_r.append(mean_r)
    all_g.append(mean_g)
    all_b.append(mean_b)

    # Brightness (Luminance)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_gray = gray.mean()
    all_brightness.append(mean_gray)

    # RMS Contrast
    rms_contrast = gray.std()
    all_contrast.append(rms_contrast)

    print(f"FILE: {img_name}")
    print(f"  - RGB: R={mean_r:.2f}, G={mean_g:.2f}, B={mean_b:.2f}")
    print(f"  - Brightness: {mean_gray:.2f} | Contrast: {rms_contrast:.2f}")
    print("-" * 30)

# ---------------------------------------------------------
# STAGE 3: FINAL SUMMARY (MIN AND MAX VALUES)
# ---------------------------------------------------------

if selected_samples:
    print("\n" + "="*50)
    print("RANGE ANALYSIS: MINIMUM AND MAXIMUM VALUES")
    print("="*50)
    
    print(f"Red Intensity:   Min={min(all_r):.2f} | Max={max(all_r):.2f}")
    print(f"Green Intensity: Min={min(all_g):.2f} | Max={max(all_g):.2f}")
    print(f"Blue Intensity:  Min={min(all_b):.2f} | Max={max(all_b):.2f}")
    print("-" * 50)
    print(f"Brightness:      Min={min(all_brightness):.2f} | Max={max(all_brightness):.2f}")
    print(f"RMS Contrast:    Min={min(all_contrast):.2f} | Max={max(all_contrast):.2f}")
    print("="*50)
