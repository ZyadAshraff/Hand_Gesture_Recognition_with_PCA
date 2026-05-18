# Hand Gesture Recognition with PCA

A machine learning pipeline that classifies hand gestures from infrared images using HOG feature extraction, PCA dimensionality reduction, and multiple classical classifiers — with an interactive Gradio demo.

---

## Overview

This project tackles hand gesture recognition without deep learning, relying instead on a carefully designed feature engineering pipeline:

1. **Preprocessing** — grayscale conversion, aspect-ratio-preserving resize, CLAHE enhancement, denoising, and normalization
2. **Segmentation** — Otsu thresholding + morphological cleanup to isolate the hand
3. **Feature Extraction** — Histogram of Oriented Gradients (HOG) on the segmented region
4. **Dimensionality Reduction** — PCA retaining 95% of variance
5. **Classification** — SVM, Random Forest, and KNN, with 5-fold cross-validation
6. **Interactive Demo** — Gradio web UI for uploading and predicting gestures in real time

---

## Dataset

**[LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)** from Kaggle (GTI-UPM)

- Infrared images of 10 distinct hand gesture classes
- Downloaded via `kagglehub`

---

## Project Structure

```
Hand_Gesture_Recognition_with_PCA.ipynb   # Main notebook
README.md
```

---

## Pipeline Details

### Preprocessing
- Resize with black padding to 128×128 (preserves aspect ratio)
- `fastNlMeansDenoising` for noise removal
- CLAHE (clipLimit=2.0, tileGrid=8×8) for contrast enhancement
- Gaussian blur (3×3) for smoothing
- Normalize pixel values to [0, 1]

### Segmentation
- Otsu's binary thresholding
- Morphological closing → opening with an elliptical 5×5 kernel

### Feature Extraction
- HOG: 8 orientations, 8×8 px/cell, 2×2 cells/block, L2-Hys normalization
- Applied on 48×48 resized segmented images
- Multi-threaded with `ThreadPoolExecutor` (8 workers)

### Dimensionality Reduction
- `StandardScaler` normalization before PCA
- PCA component count determined automatically at 95% cumulative explained variance

### Classifiers
| Model | Key Hyperparameters |
|---|---|
| SVM | RBF kernel, C=10, gamma='scale' |
| Random Forest | 50 estimators |
| KNN | k=5 |

---

## Requirements

```bash
pip install numpy pandas opencv-python scikit-learn scikit-image matplotlib gradio kagglehub
```

---

## Usage

1. Open the notebook in Jupyter or Kaggle
2. Run all cells sequentially
3. The final cell launches a **Gradio interface** — upload any hand gesture image and get an instant prediction showing the original image, segmented hand, and predicted class

---

## Results

The notebook outputs:
- Per-model test accuracy and classification reports
- Confusion matrices for all three classifiers
- 5-fold cross-validation mean ± std for each model
- A bar chart comparing model accuracies
- Demo predictions on one sample per gesture class

The best-performing model is automatically selected for the Gradio demo.

---

## License

For educational and research use. Dataset license follows the [LeapGestRecog Kaggle terms](https://www.kaggle.com/datasets/gti-upm/leapgestrecog).
