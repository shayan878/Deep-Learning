#MRI Segmentation with U-Net

This notebook (`U-Net_MRI_Segmentation.py`) implements a **U-Net convolutional neural network** for **medical image segmentation** on an MRI dataset.  
The main goal is to train a segmentation model that can accurately detect and segment brain regions/tumors from MRI scans.

---

## 📘 Purpose of the code
- Preprocess MRI images and their corresponding masks  
- Apply **data augmentation** (flips, brightness/contrast changes, elastic transforms, noise) to improve generalization  
- Implement a **U-Net architecture** in PyTorch  
- Train the model and evaluate performance using **Dice Coefficient** and **Intersection-over-Union (IoU)**  
- Visualize predictions compared to ground truth masks  

---

## ⚙️ Logic of the code
1. **Dataset preparation**
   - Load image/mask paths from CSV
   - Organize into training/validation/test splits (80% / 10% / 10%)
   - Normalize and resize images to `128×128`

2. **Data augmentation**
   - Horizontal & vertical flips  
   - Random brightness/contrast adjustment  
   - Elastic transformations  
   - Noise injection (Gaussian, salt-and-pepper)  

3. **Model architecture: U-Net**
   - **Encoder**: convolution + ReLU blocks with down-sampling  
   - **Decoder**: transposed convolutions + skip connections for feature fusion  
   - **Output layer**: 1×1 convolution to produce segmentation mask  

4. **Training**
   - Optimizer: Adam (`lr=0.001`)  
   - Batch size: 32  
   - Epochs: 20  
   - Loss function: Dice + Cross-Entropy  

5. **Evaluation**
   - Metrics: Dice Coefficient & IoU Score  
   - Training, validation, and test performance tracked over epochs  
   - Sample predictions visualized against ground truth masks  

---

## 🚀 Quick start

### Run in Google Colab
1. Upload the notebook.  
2. Switch runtime to **GPU**.  
3. Run all cells.

### Run locally
```bash
# Create and activate environment
conda create -n unet_mri python=3.10 -y
conda activate unet_mri

# Install dependencies
pip install torch torchvision albumentations opencv-python numpy matplotlib scikit-learn
Then:
bash
Copy code
jupyter notebook U-Net_MRI_Segmentation.py

📊 Expected results
 - Decreasing training/validation loss curves over epochs
 - Dice and IoU scores improving with training (model learns to overlap predicted vs. ground truth regions)
 - Visual outputs show correct segmentation of tumors/regions in MRI scans
 - Occasional mis-segmentations on challenging cases (small or ambiguous regions)

📑 References
- Ronneberger et al. — U-Net: Convolutional Networks for Biomedical Image Segmentation
- Milletari et al. — V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
- Albumentations library for augmentation
- PyTorch documentation

📜 License
MIT License (you may modify as needed).
