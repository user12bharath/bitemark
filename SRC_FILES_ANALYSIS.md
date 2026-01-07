# 📁 Source Files Analysis - BiteMark Classification Project

## Overview
This document provides a detailed analysis of each file in the `src/` folder, explaining its purpose, key functions, and role in the overall system.

---

## 🔧 **1. utils.py**
**Purpose**: Basic utility functions for project setup and common operations

### Key Functions:
- `setup_gpu()` - Configures GPU for TensorFlow, enables memory growth
- `print_section_header()` - Formats console output with styled headers
- `get_class_weights()` - Calculates class weights for imbalanced datasets
- `ensure_directories()` - Creates required project directories

### Dependencies:
- TensorFlow, NumPy
- sklearn for class weight calculation

### Usage:
```python
from utils import setup_gpu, get_class_weights
gpu_available = setup_gpu()
weights = get_class_weights(labels)
```

**Status**: ✅ No data loading - utility functions only

---

## 🌐 **2. global_utils.py**
**Purpose**: Global configuration and advanced utilities for the entire project

### Key Components:
- `GlobalConfig` class - Centralized configuration management
- `setup_gpu()` - Enhanced GPU configuration with mixed precision
- `save_metrics()` - Save evaluation metrics to JSON
- `load_metrics()` - Load previously saved metrics
- Logging configuration

### Features:
- GPU memory optimization
- Mixed precision training support (FP16)
- Standardized metrics storage
- Project-wide constants

### Usage:
```python
from global_utils import GlobalConfig, setup_gpu
config = GlobalConfig()
setup_gpu(mixed_precision=True)
```

**Status**: ✅ No data loading - configuration and utilities only

---

## 🖼️ **3. data_preprocessing.py**
**Purpose**: Load, preprocess, and split image data for training

### Key Components:

#### `PreprocessingConfig` Class:
- Stores preprocessing parameters (image size, grayscale, normalization)
- Ensures consistency between training and inference

#### `BiteMarkPreprocessor` Class:
Main preprocessing pipeline with methods:

1. **`load_sample_data(data_dir='data/augmented')`**
   - Loads images from directory structure
   - Auto-detects class folders
   - Applies preprocessing pipeline
   - **DEFAULT**: Now loads from `data/augmented` ✅

2. **`_preprocess_image(img)`**
   - Denoising (reduces camera noise)
   - CLAHE (adaptive histogram equalization for contrast)
   - Grayscale/RGB conversion
   - Resizing with high-quality interpolation
   - Normalization to [0, 1]

3. **`split_data(images, labels)`**
   - Splits into train/val/test sets
   - Stratified splitting (preserves class distribution)
   - Default: 70% train, 10% val, 20% test

4. **`create_tf_dataset(images, labels)`**
   - Creates TensorFlow datasets
   - Batching, shuffling, prefetching
   - Optional augmentation

### Preprocessing Pipeline:
```
Raw Image → Denoise → CLAHE → Grayscale → Resize → Normalize → [0,1] Float32
```

### Usage:
```python
preprocessor = BiteMarkPreprocessor(img_size=(224, 224), grayscale=True)
images, labels, class_names = preprocessor.load_sample_data(data_dir='../data/augmented')
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(images, labels)
```

**Status**: ✅ **UPDATED** - Now uses augmented data by default

---

## 🎨 **4. augmentation.py**
**Purpose**: Generate augmented training data with class balancing

### Key Components:

#### `AugmentationConfig` Class:
Configurable augmentation parameters:
- Brightness: (0.7, 1.3)
- Contrast: (0.8, 1.2)
- Rotation: ±15 degrees
- Zoom: (0.95, 1.05)
- Noise, blur, shear, perspective transforms
- Probability controls for each augmentation

#### `BiteMarkAugmentor` Class:
Advanced augmentation engine:

1. **`augment_dataset(images, labels, augmentation_factor=2)`**
   - **Intelligent Class Balancing**: Automatically balances minority classes
   - Calculates class-specific augmentation factors
   - Saves augmented images to disk (optional)
   - Reports before/after statistics

2. **`apply_random_augmentation(image)`**
   - Applies probabilistic augmentation chain
   - Preserves bite mark features (forensic-optimized)
   - Deterministic (uses fixed seed for reproducibility)

3. **Individual Augmentation Methods**:
   - `_rotate()` - Controlled rotation
   - `_flip_horizontal()` - Mirror flip
   - `_adjust_brightness()` - Lighting simulation
   - `_adjust_contrast()` - Dynamic range
   - `_add_noise()` - Camera noise
   - `_apply_blur()` - Focus variation
   - `_zoom()` - Scale transformation
   - `_adjust_saturation()` - Color enhancement
   - `_shear()` - Angle variation
   - `_perspective_transform()` - Perspective correction

### Example Output:
```
Original: dog(3), human(24), snake(35)
Augmented: dog(106), human(127), snake(115)
Total: 62 → 348 images (5.6x increase)
```

### Usage:
```python
augmentor = BiteMarkAugmentor(preserve_features=True, balance_classes=True)
aug_images, aug_labels = augmentor.augment_dataset(
    images, labels, augmentation_factor=3, 
    save_augmented=True, augmented_dir='data/augmented'
)
```

**Status**: ✅ **CORRECTLY CONFIGURED** - Loads from raw, saves to augmented

---

## 🧠 **5. enhanced_cnn.py**
**Purpose**: Define enhanced CNN architecture with advanced features

### Key Components:

#### Custom Layers:
1. **`SEBlock` (Squeeze-and-Excitation)**
   - Channel-wise attention mechanism
   - Improves feature discrimination
   - Adaptive feature recalibration

2. **`AttentionModule`**
   - Spatial attention mechanism
   - Focuses on important regions
   - Enhances bite mark feature detection

#### `EnhancedBiteMarkCNN` Class:
Advanced model builder with:

1. **Architecture Features**:
   - Depthwise separable convolutions (memory efficient)
   - Batch normalization layers
   - Dropout for regularization
   - Multiple pooling strategies
   - SE blocks for channel attention
   - Attention modules for spatial focus
   - Dense classification head

2. **Model Variants**:
   - `build_efficient_model()` - Custom lightweight CNN
   - `build_mobilenet_model()` - MobileNetV2 transfer learning
   - `build_enhanced_model()` - With SE blocks and attention

3. **Training Features**:
   - Mixed precision support
   - Custom callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
   - Class weight handling
   - TensorBoard logging

### Model Statistics:
- Parameters: ~1-5M (depending on variant)
- Input: (224, 224, 1) or (224, 224, 3)
- Output: 3 classes (dog, human, snake)
- Optimized for 4GB GPU

### Usage:
```python
model = EnhancedBiteMarkCNN(input_shape=(224, 224, 1), num_classes=3)
model.build_enhanced_model()
model.compile_model(learning_rate=0.001)
```

**Status**: ✅ No data loading - architecture definition only

---

## 🏋️ **6. train_cnn.py**
**Purpose**: Complete training pipeline from data loading to model training

### Workflow:

1. **Setup**
   - GPU configuration
   - Directory creation
   - Parameter configuration

2. **Data Loading** ✅ **UPDATED**
   ```python
   images, labels, class_names = preprocessor.load_sample_data(
       data_dir='../data/augmented'  # NOW USES AUGMENTED DATA
   )
   ```

3. **Data Splitting**
   - Train/Val/Test split with stratification
   - Reports sample counts and percentages

4. **Augmentation** (Optional)
   - Can apply additional augmentation on training data
   - Uses BiteMarkAugmentor

5. **TF Dataset Creation**
   - Batching, shuffling, caching
   - Prefetching for performance
   - Optional online augmentation

6. **Class Weights**
   - Calculates weights for imbalanced classes
   - Helps model learn minority classes

7. **Model Building**
   - Creates CNN architecture
   - Displays model summary
   - Compiles with optimizer

8. **Training**
   - Fits model with callbacks
   - Saves best model to `models/best_model.h5`
   - Logs to TensorBoard
   - Saves training info to JSON

### Configuration:
```python
IMG_SIZE = (224, 224)
GRAYSCALE = True
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
AUGMENTATION_FACTOR = 2
```

### Outputs:
- `models/best_model.h5` - Trained model
- `outputs/training_info.json` - Training metadata
- `outputs/logs/` - TensorBoard logs

**Status**: ✅ **UPDATED** - Now loads from augmented data

---

## 📊 **7. evaluate_model.py**
**Purpose**: Comprehensive model evaluation on test set

### Workflow:

1. **Load Test Data** ✅ **UPDATED**
   ```python
   images, labels, class_names = preprocessor.load_sample_data(
       data_dir='../data/augmented'  # NOW USES AUGMENTED DATA
   )
   ```

2. **Data Splitting**
   - Uses same split as training (stratified)

3. **Model Loading**
   - Loads trained model from `models/best_model.h5`
   - Handles custom layers (SE blocks, attention)

4. **Evaluation Metrics**:
   - Test loss and accuracy
   - Precision, recall, F1-score
   - Classification report (per-class metrics)
   - Confusion matrix analysis

5. **Advanced Analysis**:
   - **ROC/AUC curves** - Multi-class ROC analysis
   - **Misclassification analysis** - Visualizes errors
   - **Calibration curves** - Confidence assessment

6. **Visualizations**:
   - Confusion matrix heatmap
   - ROC curves for each class
   - Misclassified samples grid
   - Calibration plots
   - Training history curves

### Outputs:
```
outputs/
├── confusion_matrix.png
├── roc_curves.png
├── misclassified_samples.png
├── calibration_curve.png
├── metrics.json
└── classification_report.txt
```

### Usage:
```bash
python evaluate_model.py
```

**Status**: ✅ **UPDATED** - Now loads from augmented data

---

## 🔬 **8. comprehensive_evaluator.py**
**Purpose**: Advanced evaluation suite with publication-quality visualizations

### Key Features:

#### `ComprehensiveEvaluator` Class:
Professional evaluation toolkit:

1. **Model Loading**
   - Handles custom layers (SEBlock, AttentionModule)
   - Fallback loading mechanisms

2. **Prediction Pipeline**
   - Batch prediction on test set
   - Stores probabilities and true labels

3. **Metrics Calculation**:
   - `calculate_basic_metrics()` - Accuracy, precision, recall, F1
   - `calculate_auc_metrics()` - ROC AUC for multi-class
   - `calculate_confusion_matrix()` - With normalization

4. **Visualization Methods**:
   - `plot_roc_curves()` - Multi-class ROC with micro/macro average
   - `plot_confusion_matrix()` - Heatmap with per-class accuracy
   - `plot_sample_predictions()` - Grid of predictions with confidence
   - `plot_misclassified_samples()` - Error analysis visualization
   - `plot_calibration_curve()` - Confidence calibration
   - `plot_training_history()` - Training curves

5. **Comprehensive Evaluation**:
   - `comprehensive_evaluation()` - Runs all analyses
   - Saves all plots and metrics
   - Generates complete evaluation report

### Advanced Features:
- Publication-quality plots (300 DPI)
- Seaborn styling
- Color-coded predictions (green=correct, red=incorrect)
- Per-class AUC scores
- Calibration analysis per class
- Statistical significance testing

### Usage:
```python
evaluator = ComprehensiveEvaluator()
evaluator.load_model('models/best_model.h5')
metrics = evaluator.comprehensive_evaluation(
    test_dataset, class_names, output_dir='outputs/'
)
```

**Status**: ✅ No data loading - receives dataset from caller

---

## 🔄 **9. shared_preprocessing.py**
**Purpose**: Shared preprocessing for train/inference consistency

### Key Purpose:
Ensures **EXACT SAME** preprocessing is applied during:
- Training
- Validation
- Testing
- Production inference (backend API)

### `PreprocessingConfig` Class:
Comprehensive configuration:
- Image size
- Grayscale vs RGB
- Normalization
- CLAHE parameters
- Denoising strength
- Interpolation method
- Aspect ratio preservation

### `SharedPreprocessor` Class:
Production-ready preprocessing:

1. **Image Loading**:
   - `load_and_preprocess_image()` - Single image from file
   - `load_dataset()` - Complete dataset from directory
   - Handles multiple image formats (JPG, PNG, BMP, TIFF)

2. **Preprocessing Pipeline**:
   - `apply_denoising()` - Non-local means denoising
   - `apply_clahe()` - Adaptive histogram equalization
   - `ensure_channels()` - Grayscale ↔ RGB conversion
   - `resize_image()` - With aspect ratio preservation option
   - `normalize_image()` - [0, 1] normalization

3. **Batch Processing**:
   - `preprocess_batch()` - Efficient batch preprocessing
   - Error handling per image
   - Fallback to blank image on failure

### Why Important:
```
Training Preprocessing = Inference Preprocessing
         ↓                       ↓
    Model learns          Model predicts
  on processed data     on processed data
         ↓                       ↓
      ACCURATE PREDICTIONS ✅
```

If preprocessing differs → Model sees different data → Poor performance ❌

**Status**: ✅ Used by backend API and training pipeline

---

## 🧪 **10. test_data_flow.py**
**Purpose**: Automated testing suite to verify data flow

### Test Suite:

1. **Test 1: Augmented Data Exists**
   - Checks if `data/augmented/` exists
   - Counts images per class
   - Verifies total count

2. **Test 2: Preprocessor Loading**
   - Tests explicit loading from augmented directory
   - Verifies image shapes and class distribution

3. **Test 3: Preprocessor Default**
   - Tests default loading behavior
   - Verifies it uses augmented data (>100 images)

4. **Test 4: Augmentation Consistency**
   - Tests deterministic augmentation
   - Verifies same seed produces same results

5. **Test 5: Training Pipeline Compatibility**
   - Tests complete training data pipeline
   - Verifies splitting and TF dataset creation

### Test Results:
```
✅ All 5/5 tests passed
✅ Data flow properly configured
✅ 348 augmented images available
```

### Usage:
```bash
python test_data_flow.py
```

**Status**: ✅ Verification tool - all tests passing

---

## 📊 **Data Flow Summary**

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA FLOW PIPELINE                      │
└─────────────────────────────────────────────────────────────┘

data/raw/ (62 images)
    │
    │ [augmentation.py]
    ↓
data/augmented/ (348 images) ← MAIN DATA SOURCE ✅
    │
    │ [data_preprocessing.py / shared_preprocessing.py]
    ↓
Preprocessed & Split Data
    │
    ├─→ [train_cnn.py] → Train Model → models/best_model.h5
    │
    ├─→ [evaluate_model.py] → Evaluate → outputs/metrics.json
    │
    └─→ [comprehensive_evaluator.py] → Advanced Analysis → outputs/*.png
```

---

## 🎯 **File Dependencies**

```
utils.py ←────────────┐
global_utils.py ←─────┼─── All other files depend on these
                      │
data_preprocessing.py ├─── train_cnn.py
                      ├─── evaluate_model.py
                      │
augmentation.py ──────┤
                      │
shared_preprocessing.py ─── backend/app_enhanced.py
                      │
enhanced_cnn.py ──────┼─── train_cnn.py
                      ├─── evaluate_model.py
                      └─── comprehensive_evaluator.py
```

---

## ✅ **Status Summary**

| File | Purpose | Data Source | Status |
|------|---------|-------------|--------|
| utils.py | Utilities | None | ✅ Working |
| global_utils.py | Global config | None | ✅ Working |
| data_preprocessing.py | Load & preprocess | `data/augmented` | ✅ Updated |
| augmentation.py | Generate augmented data | `data/raw` → `data/augmented` | ✅ Working |
| enhanced_cnn.py | Model architecture | None | ✅ Working |
| train_cnn.py | Training pipeline | `data/augmented` | ✅ Updated |
| evaluate_model.py | Model evaluation | `data/augmented` | ✅ Updated |
| comprehensive_evaluator.py | Advanced analysis | Receives dataset | ✅ Working |
| shared_preprocessing.py | Consistent preprocessing | Configurable | ✅ Working |
| test_data_flow.py | Test suite | `data/augmented` | ✅ All passing |

---

## 🚀 **Recommended Execution Order**

1. **One-time Setup**:
   ```bash
   python augmentation.py          # Generate augmented data
   python test_data_flow.py        # Verify data flow
   ```

2. **Training**:
   ```bash
   python train_cnn.py             # Train model
   ```

3. **Evaluation**:
   ```bash
   python evaluate_model.py        # Evaluate model
   ```

4. **Production**:
   ```bash
   cd ../backend
   python app_enhanced.py          # Start API server
   ```

---

## 📈 **Current Project Status**

- ✅ **Data**: 348 balanced augmented images ready
- ✅ **Preprocessing**: Consistent pipeline configured
- ✅ **Architecture**: Enhanced CNN with attention mechanisms
- ✅ **Training Pipeline**: Ready to train
- ✅ **Evaluation Suite**: Comprehensive metrics ready
- ✅ **Backend API**: Production-ready with shared preprocessing
- ✅ **Tests**: All 5/5 data flow tests passing

**You're ready to train and deploy your bite mark classification system! 🎉**
