
# Implement Vector Quantized Variational Autoencoder (VQ-VAE) for defect detection in product images.



This project implements a Vector Quantized Variational Autoencoder (VQ-VAE) for unsupervised defect detection in product images. By training the model on only defect-free images, it learns a compact representation of normal patterns. When presented with defective images, the model fails to accurately reconstruct them — and this reconstruction error is used to detect and localize defects. The approach is powerful for quality inspection tasks in manufacturing where defective samples are rare or unlabeled.

## Model Architecture
1. Encoder
The encoder downsamples the input image using convolutional layers to extract meaningful features.
```bash
nn.Conv2d(in_channels, hidden_channels, 4, 2, 1)
→ ReLU → 
nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1)
→ ReLU → 
nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1)
```
2. Vector Quantizer
Key components: 
Codebook size: 512 , Embedding dimension: 64

Loss: 
Reconstruction loss
, Vector quantization loss (including commitment loss)
```bash
distances = ||z - e||²
quantized = nearest_embedding(z)
loss = ||z - sg(quantized)||² + β * ||sg(z) - quantized||
```
3. Decoder
```bash
nn.ConvTranspose2d(embedding_dim, hidden_channels, 3, 1, 1)
→ ReLU → 
nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1)
→ ReLU → 
nn.ConvTranspose2d(hidden_channels, 3, 4, 2, 1)
→ Sigmoid
```
## Installation

Requirements

- Python 3.7+
- PyTorch (tested with 1.10+)
- torchvision
- numpy
- OpenCV (cv2)
- Pillow
- matplotlib

Install via pip
```bash
 pip install torch torchvision numpy matplotlib pillow opencv-python

```
    
## Dataset
Download the MVTec Anomaly Detection dataset(https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it to a local directory.
Set the dataset path in the script:
```bash
 DATASET_PATH = "path_to_mvtec_dataset"

```



## Training
1. Configure hyperparameters:
You can adjust batch size, epochs, learning rate, and image resize:
```bash
batch_size = 16
epochs = 30
lr = 1e-3
resize = 256
```
2. Start training:
The script will train the VQ-VAE model on each class
```bash
model = VQVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, _, _ in train_loader:
        x = x.to(device)
        x_recon, vq_loss = model(x)
        recon_loss = torch.mean((x - x_recon) ** 2)
        loss = recon_loss + vq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss / len(train_loader):.4f}")


```
## Testing & Visualization
After training, the model is evaluated on the test set by computing pixel-wise reconstruction error. Anomalies are highlighted based on high error regions.
- Input → Reconstructed by VQ-VAE
- Original Image
- Reconstruction
- Ground Truth Mask
- Heatmap of error
- Predicted Anomaly Mask
- Overlay of anomalies
Visualization Output 

Visual results are saved in the results/{class_name} directory with multiple side-by-side views for easy comparison.
```bash
model.eval()
with torch.no_grad():
    for x, y, mask in test_loader:
        x = x.to(device)
        x_recon, _ = model(x)
        recon_error = torch.mean((x - x_recon) ** 2, dim=1, keepdim=True)
        norm_map = (recon_error - recon_error.min()) / (recon_error.max() - recon_error.min() + 1e-8)
        pred_mask = (norm_map > threshold).astype(np.uint8) * 255

        # Save heatmaps and overlays...


```
## Output
- For each class in the dataset, the model saves visual results in:
```bash
 results/{class_name}/

```
Each saved image contains a horizontal layout of the following:

1. Original Image
2. Reconstructed Image
3. Ground Truth Mask
4. Reconstruction Error Heatmap
5. Predicted Anomaly Mask
6. Overlay (Anomalies highlighted in red)

## Example Output File:

```bash 
results/bottle/bottle_005_viz.png

```


## Conclusion
This project presents an unsupervised anomaly detection pipeline using a Vector Quantized Variational Autoencoder (VQ-VAE) trained on the MVTec Anomaly Detection dataset. The model is designed to learn the distribution of normal images by minimizing reconstruction error. During testing, it detects anomalies by identifying regions with high pixel-wise reconstruction error. This allows for effective localization of defects without needing anomaly labels during training. The approach is simple yet powerful, making it a strong baseline for industrial inspection tasks. Clear visual outputs further aid in interpreting the model’s predictions and validating its performance.