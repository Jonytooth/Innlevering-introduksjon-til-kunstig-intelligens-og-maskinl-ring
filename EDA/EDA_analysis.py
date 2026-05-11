import cv2
import numpy as np
import os

IMG_DIR = r"Legg inn hvor bildene ligger"

images = [
    "dataset01_12_00019449.jpg",
    "dataset02_03_00004677.jpg",
    "dataset03_01_00002253.jpg",
    "dataset04_15_00021690.jpg",
    "dataset05_04_0004011.jpg",
    "dataset06_05_00006330.jpg",
    "dataset07_06_00007239.jpg",
    "dataset08_01_00001785.jpg",
    "dataset09_07_00008247.jpg",
    "dataset10_04_00005817.jpg",
    "dataset11_12_00016251.jpg",
    "dataset12_10_00017742.jpg",
    "dataset13_10_00014700.jpg",
    "dataset14_02_00003006.jpg",
    "dataset15_08_00008283.jpg",
    "dataset16_06_00009315.jpg",
    "dataset17_01_00002457.jpg",
    "dataset18_01_00002406.jpg",
    "dataset19_01_00001740.jpg",
    "dataset20_05_00006897.jpg",
]

print("# Kontrastanalyse\n")

for img_name in images:
    path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(path)

    if img is None:
        print(f"- Kunne ikke lese {img_name}\n")
        continue

    # RGB-intensitet
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean_r, mean_g, mean_b = img_rgb.mean(axis=(0, 1))

    # Lysintensitet
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_gray = gray.mean()

    # RMS-kontrast
    rms_contrast = gray.std()

    print(f"= {img_name}")
    print(f"- RGB-intensitet: R={mean_r:.2f}, G={mean_g:.2f}, B={mean_b:.2f}")
    print(f"- Lysintensitet: {mean_gray:.2f}")
    print(f"- Kontrast (RMS): {rms_contrast:.2f}\n")
