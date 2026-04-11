# 🔍 GradCAM – Gradient-weighted Class Activation Mapping

> Visual explanations for CNN predictions using Gradient-weighted Class Activation Mapping (GradCAM). This project trains a convolutional neural network on an image classification task and visualizes *where* the model is looking to make its decisions.



## 📖 Overview

This project implements **GradCAM (Gradient-weighted Class Activation Mapping)** to interpret and visualize the decision-making regions of a CNN classifier. GradCAM uses the gradients flowing into the final convolutional layer to produce a coarse localization heatmap highlighting important regions in the image for predicting a concept.

GradCAM is model-agnostic to architecture and requires **no architectural changes** — it works as a post-hoc visualization technique on any CNN.

---


## 🧠 Model Architecture

The model is a **custom CNN** built with Keras/TensorFlow. Below is the current architecture:

| Layer | Type | Details |
|-------|------|---------|
| 1 | Conv2D | 32 filters, 3×3, ReLU |
| 2 | MaxPooling2D | 2×2 |
| 3 | Conv2D | 64 filters, 3×3, ReLU |
| 4 | MaxPooling2D | 2×2 |
| 5 | Conv2D | 128 filters, 3×3, ReLU |
| 6 | MaxPooling2D | 2×2 |
| 7 | Dense | 256 units, ReLU |
| 8 | Output | Softmax (N classes) |


**GradCAM target layer:** Last `Conv2D` layer (Layer 5 above).

---

## ⚙️ Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Epochs | 30 |
| Batch Size | 32 *(update if different)* |
| Input Size | 224×224 *(update if different)* |
| Train/Val Split | 80/20 *(update if different)* |
| Framework | TensorFlow / Keras |
| Platform | Google Colab |


---

## 📊 Current Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~99% |
| Validation Accuracy | ~82–84% |
| Training Loss | Very low |
| Validation Loss | Higher (diverging) |
| Epochs Trained | 30 |

### 📉 Training Curve Observations

The model shows clear signs of **overfitting**:

- Training accuracy reaches **~99%** while validation plateaus at **~82–84%**
- A growing **~15–17% generalization gap** indicates the model has memorized training data
- Validation loss likely increases or stagnates after ~15–20 epochs while training loss continues to drop

```
Epoch 1–10:   Train ↑↑  |  Val ↑↑   → Both improving (healthy)
Epoch 10–20:  Train ↑↑  |  Val →     → Divergence begins
Epoch 20–30:  Train →99% |  Val ~82%  → Clear overfitting
```

---

## 🐛 Known Issues & Limitations

- **Overfitting:** Large gap between train (99%) and validation (82–84%) accuracy
- **No regularization** applied yet in the current version
- **No early stopping** — model trains for all 30 epochs regardless of validation plateau
- Architecture may benefit from batch normalization
- Limited data augmentation pipeline

---

## 🗺️ Roadmap – Planned Improvements

The following regularization and optimization techniques are planned to close the generalization gap:

### ✅ Phase 1 – Regularization (Next Step)
- [ ] **Dropout layers** — Add `Dropout(0.3–0.5)` after Dense layers
- [ ] **L2 Weight Regularization** — Add `kernel_regularizer=l2(0.001)` to Conv and Dense layers
- [ ] **Early Stopping** — Stop training when `val_loss` stops improving (patience=5)
- [ ] **ReduceLROnPlateau** — Reduce learning rate when validation plateaus

### 🔄 Phase 2 – Data & Augmentation
- [ ] Expand augmentation: shear, brightness shift, channel shift
- [ ] Collect more training samples if possible
- [ ] Apply **MixUp** or **CutMix** augmentation

### 🏗️ Phase 3 – Architecture Improvements
- [ ] Add **Batch Normalization** after Conv layers
- [ ] Experiment with **Transfer Learning** (VGG16, ResNet50, EfficientNetB0)
- [ ] Try **Global Average Pooling** instead of Flatten to reduce parameters

### 📈 Phase 4 – Evaluation
- [ ] Add confusion matrix and per-class metrics
- [ ] Plot and save training/validation curves
- [ ] Log experiments with W&B or TensorBoard

---

## 🌡️ GradCAM Visualization

GradCAM generates a **heatmap** overlaid on the input image, highlighting the regions that most influenced the model's classification decision.

### How GradCAM Works (briefly):

1. Forward pass the image through the CNN
2. Get the output of the last convolutional layer
3. Compute gradients of the target class score with respect to that layer
4. Weight each feature map channel by its average gradient
5. Apply ReLU and resize the heatmap to the input image size
6. Overlay the heatmap on the original image

### Sample Output

```
Input Image → CNN → Prediction: "Class X" (confidence: 94%)
                                    ↓
                           GradCAM Heatmap
                    [Red = high attention region]
                    [Blue = low attention region]
```

> Add your GradCAM output images here once generated.

---

## 🚀 How to Run

### 1. Open in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qje1Bwow4PU7PnvvJzdIn0Vjqc7Nbq13)

### 2. Install Dependencies

```bash
pip install tensorflow numpy matplotlib opencv-python
```

### 3. Run Training

Execute all cells in order:
1. Data loading & preprocessing
2. Model definition
3. Model training (30 epochs)
4. GradCAM visualization

### 4. Generate GradCAM Heatmaps

```python
# Pseudocode — see notebook for actual implementation
gradcam = GradCAM(model, target_layer="last_conv_layer")
heatmap = gradcam.compute(image, class_idx)
overlay = gradcam.overlay(image, heatmap)
```

---

## 📦 Requirements

```
tensorflow >= 2.x
numpy
matplotlib
opencv-python
scikit-learn
```


> 📌 *This README is a living document. Update architecture details, results, and the changelog as the model evolves.*
