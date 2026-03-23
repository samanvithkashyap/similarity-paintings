# Painting Similarity Retrieval — ArtExtract GSoC 2026

## Why This Project

If a model can understand artistic style well enough to classify it, the natural next question is: *can it find paintings that look like each other?* Not based on metadata — based on what the paintings actually look like. Shared brushwork, compositional rhythm, colour palette. The kind of similarity a curator notices but can't always articulate.

This repository is my implementation for the **HumanAI ArtExtract GSoC 2026 task** — a content-based retrieval system that takes any painting from the National Gallery of Art collection and surfaces the most visually similar works, without any task-specific training.

---

## What I Built

A zero-shot similarity retrieval system over **4,037 paintings** from the National Gallery of Art open-access dataset, using DINOv2 vision transformer embeddings indexed with FAISS for fast nearest-neighbour search.

### Why DINOv2 and not a supervised CNN?

Supervised CNNs learn features useful for their training objective — ImageNet classification, or style prediction. Those features aren't necessarily what you want for open-ended similarity. DINOv2, trained via self-supervised learning on large diverse image corpora, produces features that capture texture, composition, and structure without being anchored to any label space. For art retrieval — where "similar" is inherently subjective and spans style, period, subject matter, and palette — that generality is exactly what you need.

The CLS token of a ViT-B/14 gives you a 768-dimensional summary of the entire image, but unlike a CNN's global average pool, it's informed by patch-level self-attention across the full canvas. Stylistic similarity tends to be a global signal, and DINOv2's attention mechanism captures it naturally.

### Why FAISS?

Brute-force cosine similarity over 4,037 embeddings would work fine at this scale, but FAISS's `IndexFlatIP` gives exact inner-product search with a clean interface for scaling later. When the collection grows — or when this gets ported to a larger dataset — switching to `IndexIVFFlat` or HNSW is a one-line change.

---

## Architecture

```
NGA Paintings (4,037 images)
        │
        ▼
DINOv2 ViT-B/14  (518×518, ImageNet normalisation)
        │  [CLS token → 768-dim embedding, no fine-tuning]
        ▼
L2 Normalisation
        │
        ▼
FAISS IndexFlatIP  (cosine similarity via inner product on unit vectors)
        │
        ▼
Top-K Retrieval
```

No training. No labels used at retrieval time. The entire pipeline is inference-only — embeddings are extracted once, saved to disk, and queried at runtime.

---

## Results

Evaluated on a random sample of 200 paintings. A retrieval counts as a hit if any of the top-K results share the same artist as the query.

| Metric | Score |
|--------|-------|
| R@1    | 0.315 |
| R@5    | 0.400 |
| R@10   | 0.440 |

These are **zero-shot** numbers — DINOv2 has never seen an art dataset. The fact that it retrieves a same-artist painting in the top 10 nearly half the time, with no supervision, reflects how much stylistic signal survives self-supervised pretraining. Fine-tuning on art data or adding LoRA adapters would push these numbers considerably higher.

Artist label is used as a proxy for visual similarity here. It's an imperfect proxy — same artist doesn't always mean visually similar, and visually similar paintings can come from different artists — but it's the cleanest ground truth available in this dataset without manual annotation.

---

## Dataset

| Detail | Value |
|--------|-------|
| Source | National Gallery of Art open-access collection |
| Images | 4,037 paintings (JPEG) |
| Metadata | `objects.csv`, filtered to `classification == 'Painting'` |
| Eval labels | `attribution` (artist name) |

---

## Requirements

```
torch
torchvision
faiss-cpu
numpy
pandas
Pillow
tqdm
matplotlib
```

Install:
```bash
pip install faiss-cpu torch torchvision numpy pandas Pillow tqdm matplotlib
```

---

## Usage

The notebook is self-contained and runs on Kaggle (GPU T4).

1. Attach the `nga-portraits` image dataset and `objects.csv` metadata dataset
2. Run cells top to bottom — embeddings are extracted and saved automatically
3. Use `query(image_path, top_k=10)` to retrieve similar paintings for any image
4. Run `evaluate_recall_at_k()` to reproduce the benchmark numbers

Embeddings are saved to `/kaggle/working/embeddings.npy` and `image_ids.npy` so you don't re-run extraction on every session.

---

## GSoC 2026 Context

This project is my second submission for the **HumanAI Foundation ArtExtract task** under GSoC 2026, complementing the [multi-task style classifier](https://github.com/samanvithkashyap/wikiart-artextract). The broader ArtExtract vision involves detecting underdrawings and hidden compositions — and retrieval is a core primitive for that: once you've identified anomalous regions in a painting, you need to find comparable works to assess whether those anomalies are stylistically typical or genuinely unusual. A model that can surface visually similar paintings from a large collection is the lookup engine that makes that comparison possible.

---

## Notebook

Full implementation: [View on Kaggle](#) <!-- add your Kaggle notebook link -->

---

## Acknowledgements

- [DINOv2 — Meta AI](https://github.com/facebookresearch/dinov2)
- [National Gallery of Art Open Access](https://www.nga.gov/open-access-images.html)
- [FAISS — Meta AI](https://github.com/facebookresearch/faiss)
