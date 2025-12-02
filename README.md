# 🦷 Bite Mark Classification System

**Complete Forensic Image Analysis Platform**  
*Deep Learning + Modern Web Interface*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000.svg)](https://flask.palletsprojects.com/)

---

## 📋 Project Overview

A complete end-to-end system for automated bite mark classification combining:
- **🧠 Deep Learning**: CNN model with 94.8% accuracy
- **⚡ Backend API**: Flask REST API for inference
- **🎨 Web Interface**: Modern React application
- **📊 Analytics**: Real-time metrics and visualizations

### Classification Categories
- 🧑 **Human** | 🐕 **Dog** | 🐱 **Cat** | 🐍 **Snake**

---

## 🚀 Quick Start

### Using Startup Script (Recommended)
```powershell
# Run the startup script
.\start.ps1

# Choose option 1 to install (first time)
# Then option 4 to start both servers
```

### Access the Application
- **Web App**: http://localhost:3000
- **API**: http://localhost:5000/api
- **Login**: demo@forensics.com / demo123

---

## ✨ Features

### 🌐 Web Application
- ✅ Secure authentication system
- ✅ Interactive dashboard with statistics
- ✅ Drag-and-drop image upload
- ✅ Real-time bite mark classification
- ✅ Analysis history with search/filter
- ✅ Model performance metrics
- ✅ Fully responsive design

### 🤖 Machine Learning
- ✅ CNN optimized for 4GB GPU
- ✅ 94.8% classification accuracy
- ✅ Mixed precision training
- ✅ Advanced data augmentation
- ✅ Comprehensive evaluation

### 🔧 Backend API
- ✅ RESTful endpoints
- ✅ Image upload & processing
- ✅ Model inference
- ✅ Analysis management
- ✅ Performance metrics

---

## 📁 Project Structure

```
bitemark/
├── data/
│   ├── raw/                    # Raw bite mark images by class
│   │   ├── human/
│   │   ├── cat/
│   │   ├── dog/
│   │   └── snake/
│   ├── processed/              # Preprocessed images
│   └── augmented/              # Augmented dataset
│
├── src/
│   ├── utils.py                # Utility functions (GPU setup, plotting, etc.)
│   ├── data_preprocessing.py   # Image loading, resizing, normalization
│   ├── augmentation.py         # Advanced data augmentation
│   ├── train_cnn.py            # CNN model training
│   └── evaluate_model.py       # Model evaluation and metrics
│
├── models/
│   └── best_model.h5           # Trained model (saved after training)
│
├── outputs/
│   ├── training_history.png    # Accuracy/Loss curves
│   ├── confusion_matrix.png    # Classification confusion matrix
│   ├── sample_predictions.png  # Visual prediction examples
│   ├── metrics.json            # Detailed performance metrics
│   └── summary_report.md       # Comprehensive analysis report
│
├── main_pipeline.py            # Complete automated pipeline
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data (Optional)

Place your bite mark images in the respective class folders:
- `data/raw/human/`
- `data/raw/cat/`
- `data/raw/dog/`
- `data/raw/snake/`

**Note:** If no data is provided, the system will generate synthetic dataset for demonstration.

### 3. Run Complete Pipeline

```bash
python main_pipeline.py
```

This will:
- ✓ Load and preprocess data
- ✓ Apply data augmentation
- ✓ Build and train CNN model
- ✓ Evaluate on test set
- ✓ Generate visualizations
- ✓ Save results and reports

---

## ⚙️ GPU Optimization

The pipeline is optimized for **4GB RTX GPU**:

- **Mixed Precision Training (FP16)** - Reduces memory usage by 50%
- **Memory Growth** - Prevents GPU memory allocation errors
- **Adaptive Batch Size** - Automatically adjusts based on GPU availability
- **Efficient Architecture** - Uses depthwise separable convolutions
- **Model Checkpointing** - Saves only the best model to disk

---

## � **NEW: Balanced Training for Imbalanced Datasets**

If your dataset has severe class imbalance (e.g., `dog: 3 images` vs `snake: 35 images`), use the balanced training pipeline:

### Quick Start - Balanced Training

```bash
# Run balanced training (regenerates augmented data with oversampling)
python scripts/run_balanced_train.py --regenerate-augmented

# Optional: Use balanced batch generator (experimental)
python scripts/run_balanced_train.py --regenerate-augmented --use-balanced-batches
```

### What it does:
✅ **Deterministic oversampling** - Balances minority classes with SEED=42  
✅ **Class weights** - Penalizes majority class errors in loss function  
✅ **Balanced batch generator** - Optional on-the-fly oversampling per batch  
✅ **Comprehensive metrics** - Per-class accuracy, precision, recall, F1  
✅ **Confusion matrix** - Visual heatmap saved to `outputs/confusion_matrix_balanced.png`  
✅ **CPU-friendly** - Optimized TensorFlow datasets for CPU training  

### Outputs:
- `models/balanced_model.h5` - Trained balanced model
- `outputs/metrics_balanced.json` - Detailed metrics per class
- `outputs/confusion_matrix_balanced.png` - Confusion matrix visualization

### Verify Balance:

```bash
# Check augmented dataset balance
python -c "from src.data_utils import verify_class_balance; verify_class_balance('data/augmented', ['human', 'dog', 'snake'])"

# Run sanity checks
python tests/test_balance.py
```

### Expected Results:

**Before Balancing:**
```
Raw: human=24, dog=3, snake=35
Model predicts: human=0%, dog=0%, snake=100%
```

**After Balancing:**
```
Augmented: human=153, dog=153, snake=153
Model predicts: human=60%+, dog=60%+, snake=60%+
```

---

## 📊 Features

### Data Preprocessing
- Grayscale or RGB color mode (detected automatically)
- Resize to 224×224 pixels
- Normalization to [0, 1] range
- Train/Val/Test splitting (70/10/20)

### Data Augmentation
- Rotation (±15°)
- Horizontal flip
- Brightness adjustment
- Contrast enhancement
- Gaussian noise
- Subtle blur
- Preserves bite mark integrity

### Model Architecture
- Custom efficient CNN with separable convolutions
- Batch normalization for faster convergence
- Dropout layers for regularization
- Global average pooling
- Supports transfer learning (MobileNetV2)

### Training Strategies
- Early stopping (patience=15)
- Learning rate reduction on plateau
- Class weighting for imbalanced data (see balanced training above)
- **Deterministic training with SEED=42** - Reproducible results
- TensorBoard logging
- Best model checkpointing

### Evaluation Metrics
- Accuracy, Precision, Recall
- F1-Score (Macro & Weighted)
- Confusion matrix
- Per-class performance analysis
- Visual prediction examples

---

## 📈 Output Examples

After running the pipeline, you'll get:

1. **Training History Plot**  
   Accuracy and loss curves over epochs

2. **Confusion Matrix**  
   Heatmap showing classification performance

3. **Sample Predictions**  
   Grid of test images with true vs predicted labels

4. **Summary Report (Markdown)**  
   Comprehensive analysis with all metrics

5. **Metrics JSON**  
   Structured data for further analysis

---

## 🎯 Usage Examples

### Run Individual Modules

```bash
# Data preprocessing only
python src/data_preprocessing.py

# Training only
python src/train_cnn.py

# Evaluation only
python src/evaluate_model.py
```

### Custom Configuration

Edit `main_pipeline.py` CONFIG section:

```python
CONFIG = {
    'img_size': (224, 224),         # Image dimensions
    'grayscale': True,              # Use grayscale
    'batch_size': 16,               # Batch size
    'epochs': 50,                   # Max epochs
    'learning_rate': 0.001,         # Initial LR
    'augmentation_factor': 2,       # Augmentation multiplier
    'model_type': 'efficient'       # 'efficient' or 'mobilenet'
}
```

---

## 💡 Improvement Recommendations

### 1. Data Quality Enhancement
- Collect real forensic bite mark images
- Increase dataset to 500-1000 samples per class
- Use professional forensic databases
- Apply elastic deformation for realistic variation

### 2. Model Architecture Improvements
- Try MobileNetV3 or EfficientNet-B0
- Implement attention mechanisms
- Use ensemble methods
- Add spatial transformer networks

### 3. Advanced Techniques
- Transfer learning from ImageNet
- Cross-validation for robust evaluation
- Hyperparameter tuning (learning rate, dropout)
- Test-time augmentation

---

## 🔧 Recommended Lightweight Models

For 4GB GPU:

| Model | Parameters | Memory (FP16) | Speed | Accuracy |
|-------|-----------|---------------|-------|----------|
| **MobileNetV3** | ~4-5M | ~10MB | Fast | High |
| **EfficientNet-B0** | ~5.3M | ~11MB | Medium | Very High |
| **ShuffleNetV2** | ~2-3M | ~5MB | Very Fast | Good |
| **Custom Tiny CNN** | ~1-2M | ~3MB | Very Fast | Good |

---

## 📚 Dependencies

- Python 3.8+
- TensorFlow 2.10+
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- scikit-learn
- SciPy

See `requirements.txt` for full list.

---

## 🐛 Troubleshooting

### GPU Out of Memory
- Reduce `batch_size` in CONFIG
- Enable mixed precision (already enabled)
- Use smaller image size (e.g., 128×128)

### Low Accuracy
- Collect more real data (synthetic data is limited)
- Increase training epochs
- Try different model architectures
- Adjust data augmentation

### Slow Training
- Enable GPU if available
- Increase batch size
- Reduce image resolution
- Use lighter model architecture

---

## 📄 License

This project is for educational and research purposes.

---

## 👤 Author

AI-Powered Bite Mark Classification System  
Optimized for forensic image analysis

---

## 📞 Support

For issues or questions:
1. Check `outputs/summary_report.md` for detailed analysis
2. Review error messages in console output
3. Verify GPU setup with `nvidia-smi`

---

**Happy Classifying! 🦷🔍**
