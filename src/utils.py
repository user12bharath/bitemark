"""
Utility functions for Bite Mark Classification Pipeline
Optimized for 4GB RTX GPU
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
from datetime import datetime


def setup_gpu():
    """Configure GPU memory growth to prevent OOM errors on 4GB RTX"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU Found: {len(gpus)} device(s)")
            print(f"  GPU Details: {gpus[0].name}")
            
            # Enable mixed precision for better memory efficiency
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✓ Mixed Precision (FP16) enabled for memory optimization")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("⚠ No GPU found. Running on CPU.")
    
    return len(gpus) > 0


def create_directories():
    """Ensure all necessary directories exist"""
    dirs = [
        'data/raw/human', 'data/raw/dog', 'data/raw/snake',
        'data/processed', 'data/augmented',
        'models', 'outputs'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✓ Directory structure verified")


def plot_training_history(history, output_path='outputs/training_history.png'):
    """Plot training and validation accuracy/loss curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to {output_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, output_path='outputs/confusion_matrix.png'):
    """Plot confusion matrix heatmap"""
    import seaborn as sns  # Import only when needed
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Bite Mark Classification', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {output_path}")
    plt.close()
    
    return cm


def plot_sample_predictions(model, test_data, class_names, num_samples=12, 
                           output_path='outputs/sample_predictions.png'):
    """Display sample predictions with true vs predicted labels"""
    images, labels = next(iter(test_data.unbatch().batch(num_samples)))
    predictions = model.predict(images, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1) if len(labels.shape) > 1 else labels
    
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        img = images[i].numpy()
        if img.shape[-1] == 1:
            img = img.squeeze()
            axes[i].imshow(img, cmap='gray')
        else:
            axes[i].imshow(img)
        
        true_label = class_names[true_classes[i]]
        pred_label = class_names[pred_classes[i]]
        confidence = predictions[i][pred_classes[i]] * 100
        
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)', 
                         color=color, fontweight='bold', fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions on Test Set', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sample predictions saved to {output_path}")
    plt.close()


def export_model_summary(model, output_path='outputs/model_summary.txt'):
    """Export model architecture summary to file"""
    with open(output_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"✓ Model summary saved to {output_path}")


def plot_misclassified_samples(images, y_true, y_pred, class_names, 
                               output_path='outputs/misclassified_analysis.png',
                               num_samples=12):
    """Plot misclassified samples separately for detailed analysis"""
    wrong_indices = np.where(y_true != y_pred)[0]
    
    if len(wrong_indices) == 0:
        print("✓ No misclassified samples found!")
        return
    
    # Select random misclassified samples
    show_indices = np.random.choice(wrong_indices, min(num_samples, len(wrong_indices)), replace=False)
    
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(show_indices):
        img = images[idx]
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.squeeze()
            axes[i].imshow(img, cmap='gray')
        else:
            axes[i].imshow(img)
        
        true_class = class_names[y_true[idx]]
        pred_class = class_names[y_pred[idx]]
        
        axes[i].set_title(f'True: {true_class}\nPred: {pred_class}', 
                         fontsize=10, color='red')
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(show_indices), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Misclassified Samples Analysis ({len(wrong_indices)} total errors)', 
                 fontsize=16, fontweight='bold', color='red')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Misclassified samples saved to {output_path}")
    plt.close()
    
    return wrong_indices
    """Save evaluation metrics to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"✓ Metrics saved to {output_path}")


def print_section_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def get_class_weights(train_labels):
    """Calculate class weights for imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_classes = np.unique(train_labels)
    class_weights = compute_class_weight('balanced', 
                                         classes=unique_classes, 
                                         y=train_labels)
    class_weight_dict = dict(enumerate(class_weights))
    
    print("\n✓ Class weights calculated for imbalanced dataset:")
    for cls, weight in class_weight_dict.items():
        print(f"  Class {cls}: {weight:.3f}")
    
    return class_weight_dict


def save_metrics(metrics, output_path='outputs/metrics.json'):
    """
    Save evaluation metrics to JSON file
    
    Args:
        metrics: Dictionary of evaluation metrics
        output_path: Path to save metrics JSON
    """
    import json
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Metrics saved to: {output_path}")


def generate_summary_report(metrics, history, training_time, output_path='outputs/summary_report.md'):
    """Generate comprehensive summary report in Markdown format"""
    report = f"""# 🦷 Bite Mark Classification - Training Summary Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Dataset Statistics

- **Total Samples:** {metrics.get('total_samples', 'N/A')}
- **Training Samples:** {metrics.get('train_samples', 'N/A')}
- **Validation Samples:** {metrics.get('val_samples', 'N/A')}
- **Test Samples:** {metrics.get('test_samples', 'N/A')}
- **Classes:** {metrics.get('classes', 'N/A')}
- **Image Size:** {metrics.get('image_size', 'N/A')}

---

## ⚙️ Training Configuration

- **Training Duration:** {training_time:.2f} seconds ({training_time/60:.2f} minutes)
- **Epochs Completed:** {len(history.history['loss'])}
- **Batch Size:** {metrics.get('batch_size', 'N/A')}
- **Optimizer:** {metrics.get('optimizer', 'Adam')}
- **Learning Rate:** {metrics.get('learning_rate', 'N/A')}
- **GPU Acceleration:** {metrics.get('gpu_used', 'Yes')}
- **Mixed Precision:** Enabled (FP16)

---

## 🎯 Model Performance

### Final Metrics
- **Test Accuracy:** {metrics.get('test_accuracy', 0)*100:.2f}%
- **Test Loss:** {metrics.get('test_loss', 0):.4f}

### Per-Class Metrics
```
{metrics.get('classification_report', 'N/A')}
```

### Training Progress
- **Best Validation Accuracy:** {max(history.history.get('val_accuracy', [0]))*100:.2f}%
- **Final Training Accuracy:** {history.history.get('accuracy', [0])[-1]*100:.2f}%
- **Final Validation Accuracy:** {history.history.get('val_accuracy', [0])[-1]*100:.2f}%

---

## 📈 Confusion Matrix Summary

```
{metrics.get('confusion_matrix_str', 'N/A')}
```

---

## 💾 Model Artifacts

- **Best Model:** `models/best_model.h5`
- **Training History Plot:** `outputs/training_history.png`
- **Confusion Matrix:** `outputs/confusion_matrix.png`
- **Sample Predictions:** `outputs/sample_predictions.png`
- **Metrics JSON:** `outputs/metrics.json`

---

## 🚀 Hardware Utilization

- **GPU Model:** {metrics.get('gpu_model', 'N/A')}
- **Memory Optimization:** Mixed Precision (FP16)
- **Batch Size Optimization:** Adaptive based on 4GB VRAM

---

## ✅ Conclusion

The bite mark classification model has been successfully trained and evaluated.
Review the visualizations and metrics for detailed performance analysis.

"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ Summary report saved to {output_path}")
    return report
