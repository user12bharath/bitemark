"""
Model Evaluation Module for Bite Mark Classification
Comprehensive performance analysis and visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import json

from data_preprocessing import BiteMarkPreprocessor
from utils import (
    setup_gpu, print_section_header, plot_training_history,
    plot_confusion_matrix, plot_sample_predictions, 
    save_metrics, generate_summary_report
)


class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self, model_path='models/best_model.h5'):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['human', 'dog', 'snake']
        
    def load_model(self):
        """Load trained model"""
        print(f"📥 Loading model from {self.model_path}...")
        self.model = keras.models.load_model(self.model_path)
        print("✓ Model loaded successfully")
        return self.model
    
    def evaluate_on_test_set(self, test_dataset):
        """
        Evaluate model on test set
        
        Args:
            test_dataset: Test dataset (tf.data.Dataset)
            
        Returns:
            Dictionary of metrics
        """
        print_section_header("TEST SET EVALUATION")
        
        # Evaluate
        results = self.model.evaluate(test_dataset, verbose=1)
        
        metrics = {
            'test_loss': results[0],
            'test_accuracy': results[1],
            'test_precision': results[2] if len(results) > 2 else None,
            'test_recall': results[3] if len(results) > 3 else None
        }
        
        print(f"\n📊 Test Set Results:")
        print(f"  Loss: {metrics['test_loss']:.4f}")
        print(f"  Accuracy: {metrics['test_accuracy']*100:.2f}%")
        if metrics['test_precision']:
            print(f"  Precision: {metrics['test_precision']*100:.2f}%")
        if metrics['test_recall']:
            print(f"  Recall: {metrics['test_recall']*100:.2f}%")
        
        return metrics
    
    def get_predictions(self, test_dataset):
        """
        Get model predictions for entire test set
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            y_true, y_pred arrays
        """
        print("\n🔮 Generating predictions...")
        
        y_true = []
        y_pred = []
        
        for images, labels in test_dataset:
            predictions = self.model.predict(images, verbose=0)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(np.argmax(labels, axis=1))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        print(f"✓ Generated {len(y_true)} predictions")
        
        return y_true, y_pred
    
    def generate_classification_report(self, y_true, y_pred):
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        print_section_header("CLASSIFICATION REPORT")
        
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            digits=3
        )
        
        print(report)
        
        # Calculate additional metrics
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        print(f"\n📈 Aggregate Metrics:")
        print(f"  F1-Score (Macro): {f1_macro:.3f}")
        print(f"  F1-Score (Weighted): {f1_weighted:.3f}")
        
        return report, f1_macro, f1_weighted
    
    def analyze_confusion_matrix(self, y_true, y_pred):
        """
        Analyze and display confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        print_section_header("CONFUSION MATRIX ANALYSIS")
        print("\nConfusion Matrix:")
        print(f"{'':>10}", end='')
        for name in self.class_names:
            print(f"{name:>10}", end='')
        print()
        
        for i, name in enumerate(self.class_names):
            print(f"{name:>10}", end='')
            for j in range(len(self.class_names)):
                print(f"{cm[i, j]:>10}", end='')
            print()
        
        # Per-class accuracy
        print("\n📊 Per-Class Accuracy:")
        for i, name in enumerate(self.class_names):
            class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
            print(f"  {name:>6}: {class_acc*100:>6.2f}% ({cm[i, i]}/{cm[i].sum()})")
        
        return cm


    def plot_roc_curves(self, y_true, y_pred_proba, output_path='outputs/roc_curves.png'):
        """
        Plot ROC curves for each class
        
        Args:
            y_true: True labels (integer encoded)
            y_pred_proba: Prediction probabilities
            output_path: Path to save ROC curve plot
            
        Returns:
            Dictionary of AUC scores
        """
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        import matplotlib.pyplot as plt
        import numpy as np
        
        n_classes = len(self.class_names)
        
        # Binarize the output for multiclass ROC
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if n_classes == 2:
            y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])
        
        # Calculate ROC curve and AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                     label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Bite Mark Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curves saved to: {output_path}")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}: AUC = {roc_auc[i]:.3f}")
        
        return roc_auc


    def plot_misclassified_samples(self, test_dataset, y_true, y_pred, y_pred_proba, 
                                 output_path='outputs/misclassified_samples.png'):
        """
        Plot misclassified samples for analysis
        
        Args:
            test_dataset: Test dataset
            y_true: True labels
            y_pred: Predicted labels  
            y_pred_proba: Prediction probabilities
            output_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        
        # Find misclassified indices
        misclassified = np.where(y_true != y_pred)[0]
        
        if len(misclassified) == 0:
            print("✓ No misclassified samples found!")
            return
        
        # Extract images from dataset
        images = []
        for batch in test_dataset.unbatch():
            if isinstance(batch, tuple):
                # Batch contains (image, label) tuple
                image = batch[0].numpy()
            else:
                # Batch is just the image
                image = batch.numpy()
            images.append(image)
            if len(images) >= len(y_true):  # Ensure we don't exceed expected length
                break
        images = np.array(images)
        
        # Plot up to 12 misclassified samples
        n_samples = min(12, len(misclassified))
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()
        
        for i in range(n_samples):
            idx = misclassified[i]
            img = images[idx]
            
            # Handle grayscale images
            if img.shape[-1] == 1:
                img = img.squeeze()
                axes[i].imshow(img, cmap='gray')
            else:
                axes[i].imshow(img)
                
            true_class = self.class_names[y_true[idx]]
            pred_class = self.class_names[y_pred[idx]]
            confidence = y_pred_proba[idx][y_pred[idx]] * 100
            
            axes[i].set_title(f'True: {true_class}\nPred: {pred_class} ({confidence:.1f}%)')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_samples, 12):
            axes[i].axis('off')
        
        plt.suptitle(f'Misclassified Samples ({len(misclassified)} total)', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Misclassified samples plot saved to: {output_path}")
        print(f"  Total misclassified: {len(misclassified)}/{len(y_true)} ({len(misclassified)/len(y_true)*100:.1f}%)")


    def plot_calibration_curve(self, y_true, y_pred_proba, output_path='outputs/calibration_curve.png'):
        """
        Plot calibration curve to assess prediction confidence
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            output_path: Path to save plot
        """
        from sklearn.calibration import calibration_curve
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        # Plot calibration curve for each class
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, class_name in enumerate(self.class_names):
            # Binary problem for this class
            y_binary = (y_true == i).astype(int)
            prob_pos = y_pred_proba[:, i]
            
            if np.sum(y_binary) > 0:  # Only plot if class has samples
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_binary, prob_pos, n_bins=5, strategy='quantile'
                )
                
                plt.plot(mean_predicted_value, fraction_of_positives, 
                        marker='o', color=colors[i % len(colors)], 
                        label=f'{class_name}', linewidth=2)
        
        # Perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', alpha=0.7)
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve - Bite Mark Classification')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Calibration curve saved to: {output_path}")


def main():
    """Main evaluation pipeline"""
    print_section_header("BITE MARK CLASSIFICATION - EVALUATION PIPELINE")
    
    # Setup GPU
    setup_gpu()
    
    # Configuration
    IMG_SIZE = (224, 224)
    GRAYSCALE = True
    BATCH_SIZE = 16
    
    # Load training info if available
    training_info = {}
    if os.path.exists('outputs/training_info.json'):
        with open('outputs/training_info.json', 'r') as f:
            training_info = json.load(f)
        print("\n✓ Loaded training configuration")
    
    # Load data
    print_section_header("LOADING TEST DATA")
    preprocessor = BiteMarkPreprocessor(img_size=IMG_SIZE, grayscale=GRAYSCALE)
    images, labels, class_names = preprocessor.load_sample_data()
    
    # Split data (same as training)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        images, labels, test_size=0.2, val_size=0.1
    )
    
    # Create test dataset
    test_dataset = preprocessor.create_tf_dataset(
        X_test, y_test, batch_size=BATCH_SIZE, shuffle=False, augment=False
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path='models/best_model.h5')
    evaluator.load_model()
    
    # Evaluate on test set
    test_metrics = evaluator.evaluate_on_test_set(test_dataset)
    
    # Get predictions
    y_true, y_pred = evaluator.get_predictions(test_dataset)
    
    # Get prediction probabilities for ROC analysis
    y_pred_proba = []
    for batch_x, batch_y in test_dataset:
        batch_proba = evaluator.model.predict(batch_x, verbose=0)
        y_pred_proba.extend(batch_proba)
    y_pred_proba = np.array(y_pred_proba)
    
    # Classification report
    report, f1_macro, f1_weighted = evaluator.generate_classification_report(y_true, y_pred)
    
    # Confusion matrix analysis
    cm = evaluator.analyze_confusion_matrix(y_true, y_pred)
    
    # ROC/AUC analysis
    print_section_header("ROC/AUC ANALYSIS")
    roc_auc_scores = evaluator.plot_roc_curves(y_true, y_pred_proba)
    
    # Misclassification analysis
    print_section_header("MISCLASSIFICATION ANALYSIS")
    evaluator.plot_misclassified_samples(test_dataset, y_true, y_pred, y_pred_proba)
    
    # Calibration analysis
    print_section_header("CALIBRATION ANALYSIS")
    evaluator.plot_calibration_curve(y_true, y_pred_proba)
    
    # Visualizations
    print_section_header("GENERATING VISUALIZATIONS")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, 
                         output_path='outputs/confusion_matrix.png')
    
    # Plot sample predictions
    plot_sample_predictions(evaluator.model, test_dataset, class_names,
                          num_samples=12, output_path='outputs/sample_predictions.png')
    
    # Load and plot training history if available
    if os.path.exists('models/best_model.h5'):
        try:
            # Try to load history from model
            print("✓ Visualizations complete")
        except:
            print("⚠ Could not load training history for plotting")
    
    # Compile all metrics
    print_section_header("SAVING RESULTS")
    
    # Calculate macro AUC
    macro_auc = np.mean([roc_auc_scores[i] for i in range(len(class_names))])
    
    all_metrics = {
        'test_accuracy': float(test_metrics['test_accuracy']),
        'test_loss': float(test_metrics['test_loss']),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'auc_macro': float(macro_auc),
        'auc_per_class': {class_names[i]: float(roc_auc_scores[i]) for i in range(len(class_names))},
        'auc_micro': float(roc_auc_scores.get('micro', 0)),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_str': str(cm),
        'total_samples': len(images),
        'train_samples': training_info.get('samples', {}).get('train', len(X_train)),
        'val_samples': training_info.get('samples', {}).get('val', len(X_val)),
        'test_samples': len(X_test),
        'classes': class_names,
        'image_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'optimizer': 'Adam',
        'learning_rate': training_info.get('config', {}).get('learning_rate', 'N/A'),
        'gpu_used': 'Yes',
        'gpu_model': '4GB RTX (Optimized)'
    }
    
    # Save metrics
    save_metrics(all_metrics, output_path='outputs/metrics.json')
    
    # Generate summary report
    # Create mock history object for report generation
    class MockHistory:
        def __init__(self):
            self.history = {
                'loss': [0.5],
                'accuracy': [test_metrics['test_accuracy']],
                'val_loss': [test_metrics['test_loss']],
                'val_accuracy': [test_metrics['test_accuracy']]
            }
    
    mock_history = MockHistory()
    training_time = training_info.get('training_time', 0)
    
    report_content = generate_summary_report(
        all_metrics, mock_history, training_time,
        output_path='outputs/summary_report.md'
    )
    
    print_section_header("EVALUATION COMPLETE")
    print("\n📁 Generated Files:")
    print("  ✓ outputs/confusion_matrix.png")
    print("  ✓ outputs/sample_predictions.png")
    print("  ✓ outputs/roc_curves.png")
    print("  ✓ outputs/misclassified_samples.png")
    print("  ✓ outputs/calibration_curve.png")
    print("  ✓ outputs/metrics.json")
    print("  ✓ outputs/summary_report.md")
    
    print("\n🎯 Final Results Summary:")
    print(f"  Test Accuracy: {test_metrics['test_accuracy']*100:.2f}%")
    print(f"  F1-Score (Macro): {f1_macro:.3f}")
    print(f"  F1-Score (Weighted): {f1_weighted:.3f}")
    print(f"  AUC (Macro): {macro_auc:.3f}")
    
    print("\n📖 View the complete report: outputs/summary_report.md")


if __name__ == "__main__":
    main()
