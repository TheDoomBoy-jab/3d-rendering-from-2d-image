

A deep learning pipeline that reconstructs 3D geometry of objects (cars) from a single 2D image using a **Vision Transformer (ViT) encoder** and a **voxel cross-attention decoder**, predicting a **Truncated Signed Distance Field (TSDF)** grid and extracting a 3D mesh via Marching Cubes.

---


This project tackles the challenging problem of **single-image 3D reconstruction**. Given one RGB image of an object, the model predicts its full 3D shape as a TSDF voxel grid, which is then converted into a renderable `.obj` mesh file.

**Architecture:**
- **Encoder:** `google/vit-base-patch16-224-in21k` (pretrained Vision Transformer via HuggingFace Transformers)
- **Decoder:** Custom Voxel Cross-Attention Decoder using multi-head attention
- **Output:** 3D TSDF grid → Marching Cubes → `.obj` mesh

---


```
Input Image (224×224×3)
        ↓
   ViT Encoder (patch features: 196 × 768)
        ↓
Voxel Cross-Attention Decoder
        ↓
   TSDF Grid (32×32×32)
        ↓
  Marching Cubes (skimage)
        ↓
  3D Mesh (.obj file)
```

---


```bash
pip install tf-keras
pip install "transformers<5.0.0"
pip install scikit-image plotly
pip install open3d
pip install scipy
```


---



The dataset consists of paired image-voxel samples:

| Folder | Description |
|--------|-------------|
| `/kaggle/input/.../images/` | `.jpg` / `.png` input images |
| `/kaggle/input/.../voxels/` | `.npy` ground truth voxel grids |

Files are matched by sorted order. Unmatched extras are trimmed automatically.

---



**Loss Functions:**

- **Surface-Weighted L1 Loss** — Emphasizes accuracy near the object surface (where TSDF ≈ 0), with a configurable threshold `tau`.
- **Eikonal Loss** — Regularizes the predicted field so gradients have unit norm, encouraging a valid signed distance function geometry.

**Combined Loss:**
```
total_loss = shape_loss + (eikonal_weight × eikonal_loss)
```

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Epochs | 50 |
| Eikonal Weight | 0.1 |
| TSDF Threshold (tau) | 0.1 |

---



The notebook includes:
- **2D slice visualization** of the predicted TSDF grid (horizontal & vertical cross-sections) using `matplotlib` with a seismic colormap
- **Gaussian smoothing** of slices via `scipy.ndimage` for cleaner output
- **Interactive 3D mesh rendering** using `plotly` trisurf

---



| File | Description |
|------|-------------|
| `reconstructed_car.obj` | 3D mesh extracted via Marching Cubes |
| `m4_reconstructor_weights.h5` | Saved model weights |

---



1. Set up the environment variable before imports:
```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
```

2. Build and run the model:
```python
model = build_3d_pipeline(grid_size=32, embed_dim=768)
predicted_tsdf = model(input_image, training=False)
```

3. Extract and save the mesh:
```python
from skimage import measure
verts, faces, normals, values = measure.marching_cubes(tsdf_grid, level=0.0)
# Save as .obj ...
```

---



- The ViT model is loaded with `use_safetensors=False` due to compatibility constraints on Kaggle.
- `OPEN3D_CPU_RENDERING=true` is required for Open3D to work in headless Kaggle environments.
- The model output on untrained weights produces noise — training on paired image-voxel data is required for meaningful reconstruction.

---



This project is open for research and educational use.
