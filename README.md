# EigenFaces Case Study

**Course:** MATH 373  
**Members:** Disha Khati and Tyler Soong  
---

## Project Overview

This project applies Principal Component Analysis (PCA) to headshots of USFCA's Mathematics and Data Science faculty to compute "eigenfaces" — the fundamental visual building blocks that describe the variation across all faculty faces. We explore how many eigenfaces are needed to capture most of the variability in the dataset, and represent an individual faculty member as a linear combination of these eigenfaces.

---

## Pipeline

### Step 1: Face Alignment

Raw faculty headshots vary in framing, scale, and angle, making direct pixel-level comparison meaningless. We align all images to a common reference image (index 2) using:

- **Face detection** via OpenCV's Haar cascade classifier (`haarcascade_frontalface_default.xml`) to crop each image to just the face region.
- **Facial landmark detection** via MediaPipe's Face Landmarker model, which identifies 478 keypoints per face.
- **Affine transformation** using three anchor points — left eye (landmark 33), right eye (landmark 263), and nose tip (landmark 1) — to warp each face so these features align with the reference image.

Images where the Haar cascade cannot detect a face are skipped. In our dataset, Professor Trettel's image (index 3) failed alignment due to the face detector being unable to locate a face. All successfully aligned images are saved to `aligned_headshots/` and exported as `aligned_images.npy`.

### Step 2: PCA / Eigenfaces

With all faces aligned, we perform PCA:

- Each 128×128 grayscale image is flattened into a vector of 16,384 pixel values.
- All vectors are stacked into a matrix X and mean-centered by subtracting the average ("mean") face.
- PCA is fit on the centered matrix to find the directions of maximum variance — the eigenfaces.

### Step 3: Results

**Top 4 Eigenfaces:**  
The four most important eigenfaces are visualized as grayscale images, ordered by the proportion of variance they explain.

**Proportion of variance explained by the top 4 eigenfaces:**

| Eigenface | Variance Explained |
|-----------|-------------------|
| PC1 | 24.37% |
| PC2 | 19.10% |
| PC3 | 15.66% |
| PC4 | 6.82% |
| **Total** | **65.96%** |

The top 4 eigenfaces together explain approximately **66% of the total variability** across all faculty faces.

**Linear Combination for James (index 14):**  
Any faculty member's face can be approximated as a linear combination of the eigenfaces:

`face ≈ mean_face + PC1×eigenface1 + PC2×eigenface2 + PC3×eigenface3 + PC4×eigenface4`

For James, the coefficients are:

| Component | Coefficient |
|-----------|------------|
| PC1 | -382.32 |
| PC2 | 943.42 |
| PC3 | 44.82 |
| PC4 | 397.78 |

The reconstruction using only these 4 eigenfaces is visualized alongside the original image in the notebook.

---

## Dependencies

```bash
pip3 install opencv-python mediapipe scikit-learn numpy matplotlib
```

You will also need to download the MediaPipe Face Landmarker model file [`face_landmarker.task`](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) and place it in the project directory.

---

## File Structure

```
eigenfaces/
├── eigenfaces_caseStudy.ipynb  # Main notebook
├── face_landmarker.task        # MediaPipe model file
├── usf_headshots/              # Raw input headshots
├── aligned_headshots/          # Aligned output images
└── aligned_images.npy          # Flattened grayscale image array for PCA
```
