import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def generate_gallery():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    clean_dir = os.path.join(out_dir, "p2_data", "clean_images")
    adv_dir = os.path.join(out_dir, "adv_images")

    samples = [
        ("airplane", 0), ("automobile", 5), ("bird", 12), ("cat", 7),
        ("deer", 18), ("dog", 22), ("frog", 33), ("ship", 45)
    ]

    fig, axes = plt.subplots(len(samples), 3, figsize=(9, 2.3 * len(samples)))
    
    for i, (c, idx) in enumerate(samples):
        fc = os.path.join(clean_dir, f"{c}_{idx}.png")
        fa = os.path.join(adv_dir, f"{c}_{idx}_adv.png")

        img_c = np.array(Image.open(fc).convert("RGB")) / 255.0
        img_a = np.array(Image.open(fa).convert("RGB")) / 255.0
        diff = np.abs(img_a - img_c)
        diff_mag = np.clip(diff * 10.0, 0, 1) # 10x magnified difference map

        axes[i, 0].imshow(img_c)
        axes[i, 0].set_title(f"Clean: {c}_{idx}", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_a)
        axes[i, 1].set_title(f"Adversarial: {c}_{idx}_adv", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(diff_mag)
        axes[i, 2].set_title("Perturbation (10x Magnified)", fontsize=10)
        axes[i, 2].axis("off")

    plt.suptitle("Adversarial Imperceptibility & Perturbation Map Gallery", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    gallery_path = os.path.join(out_dir, "adversarial_gallery.png")
    plt.savefig(gallery_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Gallery saved to: {gallery_path}")

if __name__ == "__main__":
    generate_gallery()
