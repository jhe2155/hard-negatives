# Hard Negative Mining for Contrastive Learning

Final project for **6.7960 Deep Learning** (MIT, Spring 2025).  

---

## Overview

Standard contrastive learning samples negatives randomly from the training batch, which tends to produce *easy* negatives — images so dissimilar to the anchor that they contribute little gradient signal. This project investigates how **negative difficulty** affects representation quality and robustness.

**Approach:**

1. Train a baseline ResNet-18 encoder with in-batch supervised contrastive loss on Animal-10N.
2. Pre-compute a full pairwise cosine distance matrix over the training set to rank negatives by difficulty for each anchor.
3. Re-train encoders using negatives drawn from 11 difficulty tiers (nearest 1–50, 51–100, …, 501–550 cross-class neighbors).
4. Evaluate each encoder using a novel **cosine-perturbation KL-divergence robustness metric**.

---

## Robustness Metric

For each test embedding **z**, we generate *d* perturbations (one per dimension), each at a fixed cosine similarity to **z** via a rotation:

&nbsp;&nbsp;&nbsp;&nbsp;**z'** = cos(θ)**z** + sin(θ)**v**⊥

where **v**⊥ is the component of a standard basis vector orthogonal to **z**.

We pass **z** and all **z'** through a linear classifier and record the **maximum KL divergence** across perturbation directions. A lower mean max-KL indicates a more stable, robust representation.

---

## Dataset

[Animal-10N](https://github.com/sungjunhong/Animal-10N) — 55,000 noisy-labelled images across 10 visually confusable animal classes, loaded via [ActiveLoop DeepLake](https://app.activeloop.ai/activeloop/animal10n-train).

Classes: lynx, guinea pig, jaguar, cat, hamster, cheetah, coyote, chimpanzee, wolf, orangutan.

---

## Repository Structure

```
train_hard_negatives.ipynb   # Main notebook (Colab)
README.md
```

---

## Running the Notebook

Designed to run on **Google Colab** (GPU recommended — T4 is sufficient).

```
1. Open train_hard_negatives.ipynb in Colab.
2. Run the Setup cell to install deeplake.
3. The notebook will gdown pre-computed weights and nearest_negative_neighbors.pkl.
   To re-compute from scratch, follow the commented instructions in each cell.
```

**Pre-trained weights** (baseline encoder, 50 epochs, latent_dim=128):
- Downloaded automatically via `gdown` in the notebook.

**Pre-computed hard negatives** (`nearest_negative_neighbors.pkl`):
- Shape: `(N_train, 500)` — top-500 nearest cross-class neighbors per image.
- Downloaded automatically via `gdown` in the notebook.

---

## Dependencies

```
torch >= 2.0
torchvision
deeplake < 4
scikit-learn
numpy
matplotlib
tqdm
```
