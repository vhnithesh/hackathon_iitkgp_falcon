# Problem 2: Hard-Label Black-Box Adversarial Attack on a Pretrained CNN

**Competition:** ML Security Hackathon 2026 — AsianHOST & IIT Kharagpur  
**Track:** Problem 2 — Black-Box Adversarial Attack on CNN  
**Approach:** Decision-based directional search → 8-bit boundary bisection → HopSkipJump-style refinement

---

## 1. Threat model

The victim CNN is used only as an oracle that returns a predicted class index:

$$\mathcal{O}(x) = \arg\max_{c} f_c(x)$$

The attack never reads the architecture, parameters, gradients, intermediate
activations, logits or probabilities. Every access goes through one function
that normalises the image with the given mean/std, calls `model(...)`, and
returns the label (`HardLabelAdversarialAttack._predict` in `attack.py`,
`HardLabelOracle.__call__` in `hsja_refine.py`).

---

## 2. Results (all 1,000 PNGs reloaded from disk)

| Measure | Result |
| :--- | :--- |
| Attack success rate | **100.00%** (1000/1000; 95% CI 99.62–100%) |
| Mean SSIM | **0.9469** (median 0.9556; 15 images below 0.8) |
| Mean PSNR | **31.17 dB** |
| Mean L2 / L∞ | 1.7971 / 0.1140 (median L∞ 27/255) |
| Logged stage-3 queries | 353,140 over 320 images (mean 1,104 per image) |

Stage-3 refinement raised mean SSIM from 0.9269 to 0.9469 and cut the number
of images below SSIM 0.8 from 63 to 15. Clean accuracy of the victim on the
supplied images is 100%, so success is measured against the true labels.

---

## 3. Method

**Stage 1: directional search** (`attack.py`). Smooth random directions are
created by bilinearly upsampling 8×8, 12×12 and 16×16 Gaussian noise to 32×32;
smooth perturbations are less visible than pixel noise. The directions are
tried at increasing L2 radii. For the shortest label-flipping directions,
binary search finds the boundary. Each candidate is rounded to `uint8`,
re-queried, and the verified candidate with the best SSIM is kept. If no
direction flips the label, the fallback interpolates toward a clean image of
another class whose hard label has first been checked.

**Stage 2: 8-bit bisection toward the clean image** (`refine_existing.py`).
This binary-searches the line between each clean image and its adversarial
image and keeps the closest `uint8` point that is still misclassified.

**Stage 3: HopSkipJump-style refinement** (`hsja_refine.py`, run by
`refine_worst.py` on the lowest-SSIM images). Starting from a boundary point
x_b, each iteration:

1. queries the labels of x_b + δ·u_i for random unit directions u_i, and
   averages the label-weighted u_i to estimate the boundary normal;
2. steps along that normal with a geometrically shrinking step until the point
   is adversarial again;
3. bisects back toward the clean image to land on the boundary closer to it.

The settings are 15 iterations with up to 60 probes per estimate. At the end,
the point is rounded to `uint8` and verified, and it replaces the stored image
only if it is still adversarial and has higher SSIM. No image can get worse.
On this run, 270 of the 320 refined images improved and 50 kept their
previous version.

---

## 4. Tiebreakers and honesty notes

- **Query count.** Stage-3 queries are logged per image in `hsja_queries.csv`.
  The stage-1/2 images in this set came from earlier versions of the
  directional search, and the 25 worst were re-searched with
  `targeted_search.py`; those queries were not logged. Running the notebook
  with `REGENERATE = True` rebuilds every image with the current pipeline and
  logs all queries.
- **Confidence change** cannot be observed through a hard-label oracle, so it
  is not reported.

---

## 5. Reproduce

```bash
python audit_adversarial.py      # full disk-reloaded audit -> audit_results.json
python generate_gallery.py       # adversarial_gallery.png
python attack.py                 # stage 1 from scratch (overwrites adv_images/)
python refine_existing.py        # stage 2
python refine_worst.py           # stage 3 on the lowest-SSIM images
```

`p2_code.ipynb` contains stages 1 and 3 inline. By default it audits the
saved images; with `REGENERATE = True` it rebuilds all 1,000 images.
