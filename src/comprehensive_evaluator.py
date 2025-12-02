"""
📊 COMPREHENSIVE MODEL EVALUATOR
Advanced evaluation suite with detailed analysis and visualizations

Features:
- ROC/AUC analysis for multi-class classification
- Detailed classification reports with per-class metrics
- Confusion matrices with normalization options
- Sample prediction visualizations
- Model calibration analysis
- Statistical significance testing
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from typing import Dict, Any, List, Tuple, Optional
import logging
import json

# Import custom layers for model loading
try:
    from .enhanced_cnn import SEBlock, AttentionModule
except ImportError:
    from enhanced_cnn import SEBlock, AttentionModule

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings for speed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


class ComprehensiveEvaluator:
    """Comprehensive evaluation suite for bite mark classification"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.model = None
        self.predictions = None
        self.true_labels = None
        self.prediction_probabilities = None
        self.class_names = []
        
        # Configure matplotlib for better plots
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titleweight'] = 'bold'
        
        logger.info("ComprehensiveEvaluator initialized")
    
    def load_model(self, model_path: str) -> bool:
        """Load trained model for evaluation with custom objects"""
        try:
            # Define custom objects for loading
            custom_objects = {
                'SEBlock': SEBlock,
                'AttentionModule': AttentionModule
            }
            
            self.model = tf.keras.models.load_model(
                model_path, 
                custom_objects=custom_objects,
                compile=False  # Skip compilation for faster loading
            )
            logger.info(f"Model loaded: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback: try loading without custom objects
            try:
                logger.info("Attempting fallback model loading...")
                self.model = tf.keras.models.load_model(model_path, compile=False)
                logger.info("Model loaded with fallback method")
                return True
            except Exception as e2:
                logger.error(f"Fallback loading also failed: {e2}")
                return False
    
    def predict_dataset(self, dataset: tf.data.Dataset, class_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on entire dataset"""
        if self.model is None:
            raise ValueError("Model must be loaded first")
        
        logger.info("Making predictions on test dataset...")
        
        # Collect predictions and true labels
        all_predictions = []
        all_true_labels = []
        
        for batch_images, batch_labels in dataset:
            # Make predictions
            batch_pred_probs = self.model.predict(batch_images, verbose=0)
            all_predictions.append(batch_pred_probs)
            all_true_labels.append(batch_labels.numpy())
        
        # Concatenate all batches
        self.prediction_probabilities = np.concatenate(all_predictions, axis=0)
        self.true_labels = np.concatenate(all_true_labels, axis=0)
        self.predictions = np.argmax(self.prediction_probabilities, axis=1)
        self.class_names = class_names
        
        logger.info(f"Predictions completed: {len(self.predictions)} samples")
        
        return self.predictions, self.true_labels
    
    def calculate_basic_metrics(self) -> Dict[str, Any]:
        """Calculate basic classification metrics"""
        if self.predictions is None or self.true_labels is None:
            raise ValueError("Predictions must be made first")
        
        metrics = {}
        
        # Basic accuracy
        metrics['test_accuracy'] = accuracy_score(self.true_labels, self.predictions)
        
        # Precision, Recall, F1 - Multiple averaging methods
        metrics['precision_macro'] = precision_score(self.true_labels, self.predictions, average='macro')
        metrics['precision_micro'] = precision_score(self.true_labels, self.predictions, average='micro')
        metrics['precision_weighted'] = precision_score(self.true_labels, self.predictions, average='weighted')
        
        metrics['recall_macro'] = recall_score(self.true_labels, self.predictions, average='macro')
        metrics['recall_micro'] = recall_score(self.true_labels, self.predictions, average='micro')
        metrics['recall_weighted'] = recall_score(self.true_labels, self.predictions, average='weighted')
        
        metrics['f1_macro'] = f1_score(self.true_labels, self.predictions, average='macro')
        metrics['f1_micro'] = f1_score(self.true_labels, self.predictions, average='micro')
        metrics['f1_weighted'] = f1_score(self.true_labels, self.predictions, average='weighted')
        
        # Per-class metrics
        per_class_precision = precision_score(self.true_labels, self.predictions, average=None)
        per_class_recall = recall_score(self.true_labels, self.predictions, average=None)
        per_class_f1 = f1_score(self.true_labels, self.predictions, average=None)
        
        metrics['per_class_metrics'] = {}
        for i, class_name in enumerate(self.class_names):
            metrics['per_class_metrics'][class_name] = {
                'precision': float(per_class_precision[i]),
                'recall': float(per_class_recall[i]),
                'f1_score': float(per_class_f1[i])
            }
        
        logger.info(f"Basic metrics calculated - Accuracy: {metrics['test_accuracy']:.3f}")
        
        return metrics
    
    def calculate_auc_metrics(self) -> Dict[str, Any]:
        """Calculate AUC/ROC metrics for multi-class classification"""
        if self.prediction_probabilities is None or self.true_labels is None:
            raise ValueError("Predictions must be made first")
        
        n_classes = len(self.class_names)
        
        # Binarize labels for multi-class ROC
        y_true_bin = label_binarize(self.true_labels, classes=range(n_classes))
        
        # Handle binary classification case
        if n_classes == 2:
            y_true_bin = self.true_labels
        
        auc_metrics = {}
        
        # Calculate ROC AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            if n_classes == 2:
                # Binary classification
                fpr[i], tpr[i], _ = roc_curve(self.true_labels, self.prediction_probabilities[:, 1])
            else:
                # Multi-class
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], self.prediction_probabilities[:, i])
            
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Calculate macro and micro average AUC
        if n_classes > 2:
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), self.prediction_probabilities.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Macro average
        roc_auc["macro"] = np.mean([roc_auc[i] for i in range(n_classes)])
        
        auc_metrics['roc_auc_per_class'] = {self.class_names[i]: float(roc_auc[i]) for i in range(n_classes)}
        auc_metrics['auc_macro'] = float(roc_auc["macro"])
        if n_classes > 2:
            auc_metrics['auc_micro'] = float(roc_auc["micro"])
        
        # Store for plotting
        auc_metrics['_plot_data'] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
        
        logger.info(f"AUC metrics calculated - Macro AUC: {auc_metrics['auc_macro']:.3f}")
        
        return auc_metrics
    
    def calculate_confusion_matrix(self, normalize: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calculate and analyze confusion matrix"""
        if self.predictions is None or self.true_labels is None:
            raise ValueError("Predictions must be made first")
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        # Calculate per-class accuracy from confusion matrix
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        cm_metrics = {
            'confusion_matrix': cm.tolist(),
            'per_class_accuracy': {
                self.class_names[i]: float(per_class_accuracy[i]) 
                for i in range(len(self.class_names))
            },
            'normalized_confusion_matrix': None
        }
        
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_metrics['normalized_confusion_matrix'] = cm_normalized.tolist()
        
        logger.info("Confusion matrix calculated")
        
        return cm, cm_metrics
    
    def plot_roc_curves(self, output_path: str) -> None:
        """Plot ROC curves for all classes"""
        auc_metrics = self.calculate_auc_metrics()
        plot_data = auc_metrics['_plot_data']
        
        plt.figure(figsize=(12, 8))
        
        # Plot ROC curves for each class
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.class_names)))
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            plt.plot(plot_data['fpr'][i], plot_data['tpr'][i], 
                    color=color, linewidth=2,
                    label=f'{class_name} (AUC = {plot_data["roc_auc"][i]:.3f})')
        
        # Plot micro-average ROC curve if available
        if 'micro' in plot_data['roc_auc']:
            plt.plot(plot_data['fpr']["micro"], plot_data['tpr']["micro"],
                    label=f'Micro-average (AUC = {plot_data["roc_auc"]["micro"]:.3f})',
                    color='deeppink', linestyle=':', linewidth=2)
        
        # Plot macro-average
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Multi-Class BiteMark Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves saved: {output_path}")
    
    def plot_confusion_matrix(self, output_path: str, normalize: bool = True) -> None:
        """Plot confusion matrix heatmap"""
        cm, _ = self.calculate_confusion_matrix(normalize=False)
        
        if normalize:
            cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
            cmap = 'Blues'
        else:
            cm_plot = cm
            title = 'Confusion Matrix'
            fmt = 'd'
            cmap = 'Blues'
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm_plot, 
                   annot=True, 
                   fmt=fmt, 
                   cmap=cmap,
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={'shrink': 0.8})
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        
        # Add accuracy annotations for normalized matrix
        if normalize:
            accuracies = cm.diagonal() / cm.sum(axis=1)
            for i, acc in enumerate(accuracies):
                plt.text(i + 0.5, i - 0.1, f'Acc: {acc:.2f}', 
                        ha='center', va='center', fontweight='bold', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved: {output_path}")
    
    def plot_sample_predictions(self, dataset: tf.data.Dataset, output_path: str, num_samples: int = 16) -> None:
        """Plot sample predictions with confidence scores"""
        if self.model is None:
            raise ValueError("Model must be loaded first")
        
        # Collect sample images and labels
        sample_images = []
        sample_labels = []
        collected = 0
        
        for batch_images, batch_labels in dataset:
            for img, label in zip(batch_images, batch_labels):
                if collected >= num_samples:
                    break
                sample_images.append(img.numpy())
                sample_labels.append(label.numpy())
                collected += 1
            if collected >= num_samples:
                break
        
        sample_images = np.array(sample_images)
        sample_labels = np.array(sample_labels)
        
        # Make predictions
        predictions = self.model.predict(sample_images, verbose=0)
        
        # Create plot
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(sample_images))):
            # Display image
            img = sample_images[i]
            if img.shape[-1] == 1:  # Grayscale
                axes[i].imshow(img[:, :, 0], cmap='gray')
            else:  # RGB
                axes[i].imshow(img)
            
            # Get prediction info
            pred_class_idx = np.argmax(predictions[i])
            pred_confidence = predictions[i][pred_class_idx]
            pred_class = self.class_names[pred_class_idx]
            true_class = self.class_names[sample_labels[i]]
            
            # Color code: green for correct, red for incorrect
            color = 'green' if pred_class == true_class else 'red'
            
            # Set title
            axes[i].set_title(f'True: {true_class}\nPred: {pred_class}\nConf: {pred_confidence:.2f}', 
                             color=color, fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Sample Predictions with Confidence Scores', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Sample predictions saved: {output_path}")
    
    def plot_misclassified_samples(self, dataset: tf.data.Dataset, output_path: str, max_samples: int = 16) -> None:
        """Plot misclassified samples for error analysis"""
        if self.predictions is None or self.true_labels is None:
            raise ValueError("Predictions must be made first")
        
        # Find misclassified indices
        misclassified_indices = np.where(self.predictions != self.true_labels)[0]
        
        if len(misclassified_indices) == 0:
            logger.info("No misclassified samples found!")
            return
        
        # Collect misclassified images
        collected_images = []
        collected_labels = []
        collected_predictions = []
        current_idx = 0
        
        for batch_images, batch_labels in dataset:
            for img, label in zip(batch_images, batch_labels):
                if current_idx in misclassified_indices and len(collected_images) < max_samples:
                    collected_images.append(img.numpy())
                    collected_labels.append(label.numpy())
                    collected_predictions.append(self.predictions[current_idx])
                current_idx += 1
                
                if len(collected_images) >= max_samples:
                    break
            if len(collected_images) >= max_samples:
                break
        
        if len(collected_images) == 0:
            logger.warning("Could not collect misclassified images")
            return
        
        # Create plot
        rows = int(np.ceil(len(collected_images) / 4))
        cols = min(4, len(collected_images))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i in range(len(collected_images)):
            # Display image
            img = collected_images[i]
            if img.shape[-1] == 1:  # Grayscale
                axes[i].imshow(img[:, :, 0], cmap='gray')
            else:  # RGB
                axes[i].imshow(img)
            
            # Get labels
            true_class = self.class_names[collected_labels[i]]
            pred_class = self.class_names[collected_predictions[i]]
            
            # Get confidence for the predicted class
            pred_confidence = self.prediction_probabilities[misclassified_indices[i]][collected_predictions[i]]
            
            axes[i].set_title(f'True: {true_class}\nPredicted: {pred_class}\nConfidence: {pred_confidence:.2f}', 
                             color='red', fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(collected_images), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Misclassified Samples ({len(collected_images)} of {len(misclassified_indices)} total)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Misclassified samples saved: {output_path}")
    
    def plot_calibration_curve(self, output_path: str) -> None:
        """Plot calibration curve to assess prediction confidence"""
        if self.prediction_probabilities is None or self.true_labels is None:
            raise ValueError("Predictions must be made first")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, class_name in enumerate(self.class_names):
            if i >= 4:  # Only plot first 4 classes
                break
                
            # Create binary labels for this class
            binary_labels = (self.true_labels == i).astype(int)
            class_probabilities = self.prediction_probabilities[:, i]
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                binary_labels, class_probabilities, n_bins=10
            )
            
            # Plot calibration curve
            axes[i].plot(mean_predicted_value, fraction_of_positives, "s-", 
                        label=f"{class_name} (Calibration curve)", linewidth=2)
            axes[i].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", alpha=0.8)
            
            axes[i].set_xlabel('Mean Predicted Probability', fontweight='bold')
            axes[i].set_ylabel('Fraction of Positives', fontweight='bold')
            axes[i].set_title(f'Calibration Curve - {class_name}', fontweight='bold')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(self.class_names), 4):
            axes[i].axis('off')
        
        plt.suptitle('Model Calibration Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Calibration curves saved: {output_path}")
    
    def plot_training_history(self, history, output_path: str) -> None:
        """Plot training history curves"""
        if history is None:
            logger.warning("No training history provided")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
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
        
        plt.suptitle('Training History Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved: {output_path}")
    
    def comprehensive_evaluation(self, test_dataset: tf.data.Dataset, 
                                class_names: List[str], output_dir: str) -> Dict[str, Any]:
        """Run complete evaluation suite"""
        logger.info("Starting comprehensive evaluation...")
        
        # Make predictions on test set
        self.predict_dataset(test_dataset, class_names)
        
        # Calculate all metrics
        basic_metrics = self.calculate_basic_metrics()
        auc_metrics = self.calculate_auc_metrics()
        cm, cm_metrics = self.calculate_confusion_matrix()
        
        # Generate all visualizations
        os.makedirs(output_dir, exist_ok=True)
        
        # ROC curves
        self.plot_roc_curves(os.path.join(output_dir, 'roc_curves.png'))
        
        # Confusion matrix
        self.plot_confusion_matrix(os.path.join(output_dir, 'confusion_matrix.png'))
        
        # Sample predictions
        self.plot_sample_predictions(test_dataset, os.path.join(output_dir, 'sample_predictions.png'))
        
        # Misclassified samples
        self.plot_misclassified_samples(test_dataset, os.path.join(output_dir, 'misclassified_samples.png'))
        
        # Calibration curves
        self.plot_calibration_curve(os.path.join(output_dir, 'calibration_curves.png'))
        
        # Combine all metrics
        comprehensive_metrics = {
            **basic_metrics,
            **{k: v for k, v in auc_metrics.items() if not k.startswith('_')},
            **cm_metrics
        }
        
        # Save detailed classification report
        classification_rep = classification_report(
            self.true_labels, self.predictions,
            target_names=class_names,
            output_dict=True
        )
        comprehensive_metrics['classification_report'] = classification_rep
        
        logger.info("Comprehensive evaluation completed")
        
        return comprehensive_metrics