# Dual Augmentation (In-Model, GPU-Native)

**AugmentedModel** is a `tf.keras.Model` wrapper that performs **image–mask–consistent data augmentation inside the model graph**, so transforms run **on-GPU** during training (and optionally during evaluation). It also lets you **blend original and augmented batches** per epoch, and can compute **sample weights after augmentation** (from the transformed masks) so the loss reflects the distribution the model actually sees.

> Works for semantic segmentation (image + mask), and for multi-input models (e.g., image + categorical/continuous features). In multi-input setups, **only the image is augmented**; auxiliary features pass through unchanged.

---

## Why this exists

- **Eliminates the PCIe bottleneck:** only the raw batch crosses PCIe; all aug ops execute on-GPU.  
- **Saves storage:** no pregenerated variants or cache directories—store originals; synthesize on the fly.  
- **Ships with the model:** the augmentation pipeline can be exported with the model for consistent train/eval behavior.  
- **Correct class weighting:** **sample weights are computed *after* augmentation** from the transformed masks.  
- **Fast & stable:** designed for `tf.function`, mixed precision, and XLA (JIT) to fuse ops efficiently.  
- **Distributed-ready:** integrates cleanly with `tf.distribute.MirroredStrategy` (multi-GPU).  
- **Transfer learning friendly:** simple freeze/unfreeze utilities and layer visibility for staged fine-tuning.  
- **MLOps-aligned:** standard Keras callbacks (EarlyStopping/Checkpoint/TensorBoard/MLflow), deterministic seeding, and SavedModel export.

---

## Augmentation engine (as implemented)

### Geometric *(label-preserving; images bilinear, masks nearest)*
- `RandomFlip(mode={'horizontal','vertical','horizontal_and_vertical'})` — bboxes supported.  
- `RandomRotation(angle=[min_deg,max_deg])` — projective rotation; bboxes supported.  
- `RandomShear(shear_angle=[min_deg,max_deg])` — projective shear; bboxes supported.  
- `RandomZoom(height_factor=(%), width_factor=(%))` — projective zoom in/out; bboxes supported.  
- `RandomJitter(jitter_range=(px_min, px_max))` — **spatial jitter** (XY translation via pad+crop); bboxes shifted.

### Photometric *(image-only)*
- `RandomBrightness(factor in [-1..1], image_intensity_range=(min,max))`  
- `RandomContrast(factor ∈ [0..1])`  
- `RandomGaussianNoise(stddev=[lo..hi])`

### Cropping
- `RandomCrop(zoom_factor=%)` — crops a random window then resizes back (no padding mode).  
- `CenterCrop(height, width)` — fixed center crop (or smart-resize up if needed).

### Other
- `SampleWeight(num_classes)` — per-pixel inverse-frequency weights computed **after augmentation**.  
- `AffineTransform(rotate, shear, zoom, ...)` — fused geometric transforms.  
- `RandomAugmentationLayer(...)` — convenience chain: **Flip → Affine → Contrast → Noise → Jitter → Brightness**.

> Masks and (optional) bounding boxes receive the **same spatial transforms** as images.  
> **Note:** Normalization is not part of these layers; do it in the input pipeline or model.

---
## Install
### Getting Started

### Prerequisites

* Python 3.10+
* pip ≥ 23.3.1

### a. Clone the Repository

```bash
git clone hhttps://github.com/anthony-iheonye/augmented_model.git
cd augmented_model
```

### b. Set Up Environment

```bash
python3 -m venv vpe_venv
source vpe_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
---
## Notebook

Explore the full workflow (data, preview, training, segmentation demo) in the notebook:

- **Dual Augmentation Demo Notebook:** `notebooks/augmented_model_demo.ipynb` *(update the path/name if different)*
