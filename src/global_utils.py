"""
🛠️ GLOBAL UTILITIES
Production-grade utilities for the BiteMark Classification System

Features:
- GPU setup and optimization
- Reproducible environment configuration
- Advanced visualization functions
- Comprehensive reporting
- Cross-platform compatibility
"""

import os
import sys
import json
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GlobalConfig:
    """Global configuration for the entire system"""
    # Reproducibility
    SEED: int = 42
    
    # GPU settings
    GPU_MEMORY_GROWTH: bool = True
    GPU_MEMORY_LIMIT: Optional[int] = 4096  # MB, None for unlimited
    
    # Visualization
    FIGURE_DPI: int = 300
    FIGURE_SIZE: Tuple[int, int] = (12, 8)
    COLOR_PALETTE: str = 'viridis'
    
    # Paths
    DEFAULT_MODEL_DIR: str = 'models'
    DEFAULT_OUTPUT_DIR: str = 'outputs'
    DEFAULT_DATA_DIR: str = 'data'
    
    # File formats
    IMAGE_EXTENSIONS: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'SEED': self.SEED,
            'GPU_MEMORY_GROWTH': self.GPU_MEMORY_GROWTH,
            'GPU_MEMORY_LIMIT': self.GPU_MEMORY_LIMIT,
            'FIGURE_DPI': self.FIGURE_DPI,
            'FIGURE_SIZE': self.FIGURE_SIZE,
            'COLOR_PALETTE': self.COLOR_PALETTE
        }


def print_section_header(title: str, char: str = "=", width: int = 80):
    """Print a formatted section header"""
    print(f"\n{char * width}")
    print(f" 🦷 {title.upper()}")
    print(f"{char * width}")


def get_reproducible_seed() -> int:
    """Get a reproducible seed"""
    config = GlobalConfig()
    return config.SEED


def setup_environment(mixed_precision: bool = True, seed: int = None) -> bool:
    """Setup reproducible environment and GPU optimization"""
    if seed is None:
        seed = get_reproducible_seed()
    
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Configure TensorFlow for deterministic operations
    tf.config.experimental.enable_op_determinism()
    
    # GPU setup
    gpu_available = setup_gpu(mixed_precision)
    
    logger.info(f"Environment setup complete:")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  GPU Available: {gpu_available}")
    logger.info(f"  Mixed Precision: {mixed_precision}")
    logger.info(f"  TensorFlow Version: {tf.__version__}")
    
    return gpu_available


def setup_gpu(mixed_precision: bool = True) -> bool:
    """Setup GPU with memory growth and mixed precision"""
    try:
        # Get GPU list
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus:
            config = GlobalConfig()
            
            # Enable memory growth for all GPUs
            for gpu in gpus:
                if config.GPU_MEMORY_GROWTH:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set memory limit if specified
                if config.GPU_MEMORY_LIMIT:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=config.GPU_MEMORY_LIMIT
                        )]
                    )
            
            # Setup mixed precision
            if mixed_precision:
                try:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    logger.info("Mixed precision policy enabled")
                except Exception as e:
                    logger.warning(f"Mixed precision setup failed: {e}")
            
            logger.info(f"GPU setup complete: {len(gpus)} GPUs configured")
            return True
            
        else:
            logger.warning("No GPU found, using CPU")
            return False
            
    except Exception as e:
        logger.error(f"GPU setup failed: {e}")
        return False


def create_directories(config) -> None:
    """Create necessary directories for the pipeline"""
    directories = [
        config.model_dir,
        config.output_dir,
        config.processed_dir,
        config.augmented_dir,
        os.path.join(config.output_dir, 'logs'),
        os.path.join(config.output_dir, 'visualizations')
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created {len(directories)} directories")


def save_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    """Save metrics to JSON file with proper formatting"""
    try:
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        converted_metrics = convert_types(metrics)
        
        # Save with proper formatting
        with open(output_path, 'w') as f:
            json.dump(converted_metrics, f, indent=2, sort_keys=True)
        
        logger.info(f"Metrics saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")


def plot_training_history(history, output_path: str, metrics: List[str] = None) -> None:
    """Plot comprehensive training history"""
    if metrics is None:
        metrics = ['accuracy', 'loss']
    
    try:
        config = GlobalConfig()
        
        # Setup plot
        fig, axes = plt.subplots(2, 2, figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Plot accuracy
        if 'accuracy' in history.history:
            axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
            if 'val_accuracy' in history.history:
                axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            axes[0, 0].set_title('Model Accuracy', fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot loss
        if 'loss' in history.history:
            axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
            if 'val_loss' in history.history:
                axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
            axes[0, 1].set_title('Model Loss', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot learning rate
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'], linewidth=2, color='orange')
            axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Plot additional metrics
        if 'top_k_categorical_accuracy' in history.history:
            axes[1, 1].plot(history.history['top_k_categorical_accuracy'], 
                          label='Training Top-K Accuracy', linewidth=2)
            if 'val_top_k_categorical_accuracy' in history.history:
                axes[1, 1].plot(history.history['val_top_k_categorical_accuracy'], 
                              label='Validation Top-K Accuracy', linewidth=2)
            axes[1, 1].set_title('Top-K Accuracy', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Top-K Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Top-K Accuracy\nNot Available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to plot training history: {e}")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         output_path: str, normalize: bool = True) -> None:
    """Plot confusion matrix with advanced formatting"""
    try:
        config = GlobalConfig()
        
        # Normalize if requested
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            cm_normalized = cm
            title = 'Confusion Matrix'
            fmt = 'd'
        
        # Create plot
        plt.figure(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)
        
        # Plot heatmap
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt=fmt, 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={'shrink': 0.8})
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        
        # Add accuracy scores
        if normalize:
            accuracies = cm.diagonal() / cm.sum(axis=1)
            for i, acc in enumerate(accuracies):
                plt.text(i + 0.5, i - 0.1, f'Acc: {acc:.2f}', 
                        ha='center', va='center', fontweight='bold', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix plot saved: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to plot confusion matrix: {e}")


def plot_roc_curves(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                   class_names: List[str], output_path: str) -> None:
    """Plot ROC curves for multi-class classification"""
    try:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        config = GlobalConfig()
        
        # Binarize labels for multi-class ROC
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        n_classes = len(class_names)
        
        # Create plot
        plt.figure(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)
        
        # Plot ROC curve for each class
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            if n_classes == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            else:
                # Multi-class
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=color, linewidth=2, 
                    label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Multi-Class Classification', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves plot saved: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to plot ROC curves: {e}")


def generate_summary_report(metrics: Dict[str, Any], output_path: str) -> None:
    """Generate comprehensive markdown summary report"""
    try:
        report_content = f"""# 🦷 BiteMark Classification System - Training Report

## 📊 Executive Summary

**Training Completed:** {time.strftime('%Y-%m-%d %H:%M:%S')}

### Key Metrics
- **Test Accuracy:** {metrics.get('test_accuracy', 'N/A') * 100:.2f}%
- **Macro F1-Score:** {metrics.get('f1_macro', 'N/A'):.3f}
- **Weighted F1-Score:** {metrics.get('f1_weighted', 'N/A'):.3f}
- **Macro AUC:** {metrics.get('auc_macro', 'N/A'):.3f}

## 🏗️ Model Architecture

- **Model Type:** {metrics.get('pipeline_config', {}).get('model_type', 'N/A')}
- **Input Shape:** {metrics.get('pipeline_config', {}).get('img_size', 'N/A')}
- **Total Parameters:** {metrics.get('total_epochs', 'N/A')}
- **Mixed Precision:** {'✓' if metrics.get('mixed_precision_used', False) else '✗'}

## 📈 Training Details

### Configuration
- **Epochs:** {metrics.get('total_epochs', 'N/A')}
- **Batch Size:** {metrics.get('pipeline_config', {}).get('batch_size', 'N/A')}
- **Learning Rate:** {metrics.get('pipeline_config', {}).get('learning_rate', 'N/A')}
- **Augmentation Factor:** {metrics.get('pipeline_config', {}).get('augmentation_factor', 'N/A')}

### Performance
- **Best Validation Accuracy:** {metrics.get('best_val_accuracy', 'N/A') * 100:.2f}%
- **Final Training Accuracy:** {metrics.get('final_train_accuracy', 'N/A') * 100:.2f}%
- **Final Validation Accuracy:** {metrics.get('final_val_accuracy', 'N/A') * 100:.2f}%

## 🔧 System Configuration

### Hardware
- **GPU Used:** {'✓' if metrics.get('gpu_used', False) else '✗'}
- **Mixed Precision:** {'✓' if metrics.get('mixed_precision_used', False) else '✗'}
- **Reproducible Seed:** {metrics.get('reproducible_seed', 'N/A')}

### Performance
- **Total Pipeline Time:** {metrics.get('total_pipeline_time', 0) / 60:.2f} minutes
- **Class Weights Used:** {'✓' if metrics.get('class_weights_used', False) else '✗'}

## 📁 Generated Artifacts

### Model Files
- `models/best_model.h5` - Production-ready trained model
- `outputs/pipeline_config.json` - Complete pipeline configuration

### Evaluation Reports
- `outputs/metrics.json` - Comprehensive metrics and results
- `outputs/summary_report.md` - This detailed report

### Visualizations
- `outputs/training_history.png` - Training and validation curves
- `outputs/confusion_matrix.png` - Classification performance matrix
- `outputs/roc_curves.png` - ROC/AUC analysis curves

## 🎯 Usage Instructions

### Production Deployment
```python
import tensorflow as tf
from src.shared_preprocessing import SharedPreprocessor

# Load the trained model
model = tf.keras.models.load_model('models/best_model.h5')

# Initialize preprocessor with same config as training
preprocessor = SharedPreprocessor()

# Process new image for prediction
processed_image = preprocessor.load_and_preprocess_image('path/to/new/image.jpg')
prediction = model.predict(processed_image)
```

### Model Performance Guidelines
- **Confidence Threshold:** Use predictions with >80% confidence
- **Batch Processing:** Optimal batch size is {metrics.get('pipeline_config', {}).get('batch_size', 16)}
- **Input Requirements:** Images should be preprocessed to {metrics.get('pipeline_config', {}).get('img_size', '(224, 224)')} pixels

## 🔍 Quality Assurance

This model has been trained using:
- ✅ Reproducible random seeds
- ✅ Advanced data augmentation
- ✅ Class imbalance handling
- ✅ Comprehensive evaluation metrics
- ✅ Production-grade preprocessing

---
*Report generated automatically by the Enhanced BiteMark Classification System*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Summary report generated: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate summary report: {e}")


def save_pipeline_state(metrics: Dict[str, Any], history: Any, config: Any, output_dir: str) -> None:
    """Save complete pipeline state for reproducibility"""
    try:
        # Save metrics
        metrics_path = os.path.join(output_dir, 'comprehensive_metrics.json')
        save_metrics(metrics, metrics_path)
        
        # Save configuration
        config_path = os.path.join(output_dir, 'pipeline_config.json')
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Save training history
        if history:
            history_path = os.path.join(output_dir, 'training_history.json')
            history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)
        
        # Generate comprehensive report
        report_path = os.path.join(output_dir, 'summary_report.md')
        generate_summary_report(metrics, report_path)
        
        logger.info(f"Pipeline state saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Failed to save pipeline state: {e}")


def log_system_info():
    """Log comprehensive system information"""
    logger.info("System Information:")
    logger.info(f"  Python Version: {sys.version}")
    logger.info(f"  TensorFlow Version: {tf.__version__}")
    logger.info(f"  Platform: {sys.platform}")
    
    # GPU information
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"  GPU Devices: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            logger.info(f"    GPU {i}: {gpu.name}")
    else:
        logger.info("  GPU Devices: None")


# Initialize global configuration
GLOBAL_CONFIG = GlobalConfig()