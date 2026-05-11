import os
import random
import shutil
from collections import defaultdict

# -----------------------------
# KONFIGURASJON – ENDRE HER
# -----------------------------
IMG_DIR = r"C:\path\to\img"     # <-- Sett riktig sti
GT_DIR  = r"C:\path\to\gt"      # <-- Sett riktig sti
OUT_DIR = r"C:\path\to\output"  # <-- Mappen der utvalgte filer skal lagres

random.seed(42)

# Opprett output-mappe hvis den ikke finnes
os.makedirs(OUT_DIR, exist_ok=True)

# Finn alle JPG-bilder
images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")]

# Grupper etter kamuflasjetype
groups = defaultdict(list)

for img in images:
    base = os.path.splitext(img)[0]  # dataset20_15_00016233
    parts = base.split("_")

    # Første del: "dataset20"
    camo_type = parts[0].replace("dataset", "")

    groups[camo_type].append(base)

# Velg én tilfeldig fra hver type og kopier
selected = {}

for camo_type, items in groups.items():
    choice = random.choice(items)

    img_src = os.path.join(IMG_DIR, choice + ".jpg")
    mask_src = os.path.join(GT_DIR, choice + ".png")

    # Sjekk at masken finnes
    if not os.path.exists(mask_src):
        print(f"ADVARSEL: Mangler maske for {choice}")
        continue

    # Kopier til OUT_DIR
    img_dst = os.path.join(OUT_DIR, choice + ".jpg")
    mask_dst = os.path.join(OUT_DIR, choice + ".png")

    shutil.copy2(img_src, img_dst)
    shutil.copy2(mask_src, mask_dst)

    selected[camo_type] = (img_dst, mask_dst)

# Print resultat
print("\nValgte og kopierte én prøve per kamuflasjetype:\n")
for camo_type, (img_path, mask_path) in sorted(selected.items(), key=lambda x: int(x[0])):
    print(f"Type {camo_type}:")
    print(f"  Bilde: {img_path}")
    print(f"  Maske: {mask_path}")
    print()
