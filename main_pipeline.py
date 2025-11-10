"""
🦷 BITE MARK CLASSIFICATION - COMPLETE PIPELINE
Automated end-to-end training and evaluation system
Optimized for 4GB RTX GPU

This script orchestrates the entire workflow:
1. Data Preprocessing
2. Data Augmentation  
3. Model Training
4. Model Evaluation
5. Result Visualization
6. Summary Report Generation
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import BiteMarkPreprocessor
from augmentation import BiteMarkAugmentor
from train_cnn import BiteMarkCNN
from evaluate_model import ModelEvaluator
from utils import (
    setup_gpu, create_directories, print_section_header,
    plot_training_history, plot_confusion_matrix, plot_sample_predictions,
    save_metrics, generate_summary_report, get_class_weights
)

import numpy as np
import tensorflow as tf


def run_complete_pipeline():
    """Execute the complete bite mark classification pipeline"""
    
    pipeline_start = time.time()
    
    # ASCII Art Header
    print("\n" + "="*80)
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║           🦷  BITE MARK CLASSIFICATION SYSTEM  🦷                        ║
    ║                                                                          ║
    ║              Deep Learning Pipeline for Forensic Analysis                ║
    ║                   Optimized for 4GB RTX GPU                              ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    print("="*80)
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    print_section_header("PHASE 0: INITIALIZATION")
    
    create_directories()
    gpu_available = setup_gpu()
    
    # Configuration - Optimized for real bite mark images
    CONFIG = {
        'img_size': (224, 224),
        'grayscale': False,  # Keep RGB for real images
        'batch_size': 16 if gpu_available else 8,
        'epochs': 100,  # More epochs for better convergence with real data
        'learning_rate': 0.0005,  # Lower LR for transfer learning
        'augmentation_factor': 3,  # Higher augmentation for imbalanced dataset
        'test_size': 0.2,
        'val_size': 0.15,  # Slightly larger validation set
        'model_type': 'mobilenet',  # Use transfer learning for better accuracy
        'adaptive_histogram': True,  # Enable CLAHE for better contrast
        'denoise': True,  # Enable denoising for real photos
        'balance_classes': True  # Balance classes during augmentation
    }
    
    print("\n⚙️  Pipeline Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key:.<25} {value}")
    
    # ========================================================================
    # PHASE 1: DATA PREPROCESSING
    # ========================================================================
    print_section_header("PHASE 1: DATA PREPROCESSING")
    
    preprocessor = BiteMarkPreprocessor(
        img_size=CONFIG['img_size'], 
        grayscale=CONFIG['grayscale'],
        adaptive_histogram=CONFIG['adaptive_histogram'],
        denoise=CONFIG['denoise']
    )
    
    print("📂 Loading dataset...")
    images, labels, class_names = preprocessor.load_sample_data(
        'data/raw', 
        save_processed=True,  # Save processed images
        processed_dir='data/processed'
    )
    
    print(f"\n✓ Dataset loaded successfully")
    print(f"  Total samples: {len(images)}")
    print(f"  Image shape: {images[0].shape}")
    print(f"  Classes: {class_names}")
    print(f"  Label distribution: {dict(zip(class_names, np.bincount(labels)))}")
    
    # Split data
    print("\n📊 Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        images, labels, 
        test_size=CONFIG['test_size'], 
        val_size=CONFIG['val_size']
    )
    
    # ========================================================================
    # PHASE 2: DATA AUGMENTATION
    # ========================================================================
    print_section_header("PHASE 2: DATA AUGMENTATION")
    
    augmentor = BiteMarkAugmentor(
        preserve_features=True,
        balance_classes=CONFIG['balance_classes']
    )
    X_train_aug, y_train_aug = augmentor.augment_dataset(
        X_train, y_train, 
        augmentation_factor=CONFIG['augmentation_factor'],
        class_names=class_names,
        save_augmented=True,  # Save augmented images
        augmented_dir='data/augmented'
    )
    
    print(f"\n✓ Augmentation summary:")
    print(f"  Original training samples: {len(X_train)}")
    print(f"  Augmented training samples: {len(X_train_aug)}")
    print(f"  Augmentation factor: {CONFIG['augmentation_factor']}x")
    
    # ========================================================================
    # PHASE 3: DATASET PREPARATION
    # ========================================================================
    print_section_header("PHASE 3: TENSORFLOW DATASET CREATION")
    
    train_dataset = preprocessor.create_tf_dataset(
        X_train_aug, y_train_aug, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        augment=True
    )
    
    val_dataset = preprocessor.create_tf_dataset(
        X_val, y_val, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        augment=False
    )
    
    test_dataset = preprocessor.create_tf_dataset(
        X_test, y_test, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        augment=False
    )
    
    print("✓ TensorFlow datasets created")
    print(f"  Training batches: ~{len(X_train_aug) // CONFIG['batch_size']}")
    print(f"  Validation batches: ~{len(X_val) // CONFIG['batch_size']}")
    print(f"  Test batches: ~{len(X_test) // CONFIG['batch_size']}")
    
    # Calculate class weights for imbalanced data
    class_weights = get_class_weights(y_train_aug)
    
    # ========================================================================
    # PHASE 4: MODEL BUILDING
    # ========================================================================
    print_section_header("PHASE 4: MODEL ARCHITECTURE")
    
    input_shape = (*CONFIG['img_size'], 3 if not CONFIG['grayscale'] else 1)
    
    cnn = BiteMarkCNN(
        input_shape=input_shape, 
        num_classes=len(class_names),
        model_type=CONFIG['model_type']
    )
    
    model = cnn.build()
    cnn.summary()
    
    cnn.compile_model(learning_rate=CONFIG['learning_rate'])
    
    # ========================================================================
    # PHASE 5: MODEL TRAINING
    # ========================================================================
    print_section_header("PHASE 5: MODEL TRAINING")
    
    callbacks_list = cnn.get_callbacks(model_path='models/best_model.h5')
    
    print(f"\n🚀 Starting training (max {CONFIG['epochs']} epochs)...")
    print("="*80)
    
    history, training_time = cnn.train(
        train_dataset, 
        val_dataset,
        epochs=CONFIG['epochs'],
        class_weights=class_weights,
        callbacks_list=callbacks_list
    )
    
    print("="*80)
    print(f"✓ Training completed in {training_time:.2f}s ({training_time/60:.2f} min)")
    
    # ========================================================================
    # PHASE 6: MODEL EVALUATION
    # ========================================================================
    print_section_header("PHASE 6: MODEL EVALUATION")
    
    evaluator = ModelEvaluator(model_path='models/best_model.h5')
    evaluator.load_model()
    
    # Evaluate on test set
    test_metrics = evaluator.evaluate_on_test_set(test_dataset)
    
    # Get predictions
    y_true, y_pred = evaluator.get_predictions(test_dataset)
    
    # Generate reports
    report, f1_macro, f1_weighted = evaluator.generate_classification_report(
        y_true, y_pred
    )
    
    cm = evaluator.analyze_confusion_matrix(y_true, y_pred)
    
    # ========================================================================
    # PHASE 7: VISUALIZATION
    # ========================================================================
    print_section_header("PHASE 7: GENERATING VISUALIZATIONS")
    
    # Plot training history
    plot_training_history(history, output_path='outputs/training_history.png')
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, 
                         output_path='outputs/confusion_matrix.png')
    
    # Plot sample predictions
    plot_sample_predictions(evaluator.model, test_dataset, class_names,
                          num_samples=12, output_path='outputs/sample_predictions.png')
    
    print("\n✓ All visualizations generated successfully")
    
    # ========================================================================
    # PHASE 8: SAVE RESULTS
    # ========================================================================
    print_section_header("PHASE 8: SAVING RESULTS")
    
    # Compile all metrics
    all_metrics = {
        'test_accuracy': float(test_metrics['test_accuracy']),
        'test_loss': float(test_metrics['test_loss']),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_str': str(cm),
        'total_samples': len(images),
        'train_samples': len(X_train_aug),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'classes': class_names,
        'image_size': CONFIG['img_size'],
        'batch_size': CONFIG['batch_size'],
        'optimizer': 'Adam',
        'learning_rate': CONFIG['learning_rate'],
        'gpu_used': 'Yes' if gpu_available else 'No',
        'gpu_model': '4GB RTX (Optimized)' if gpu_available else 'CPU'
    }
    
    # Save metrics
    save_metrics(all_metrics, output_path='outputs/metrics.json')
    
    # Generate comprehensive report
    report_content = generate_summary_report(
        all_metrics, history, training_time,
        output_path='outputs/summary_report.md'
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    pipeline_time = time.time() - pipeline_start
    
    print_section_header("🎉 PIPELINE COMPLETE")
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                         EXECUTION SUMMARY                                ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    
    ⏱️  Total Pipeline Time: {pipeline_time:.2f}s ({pipeline_time/60:.2f} minutes)
    
    📊 Dataset Statistics:
       • Total Samples: {len(images)}
       • Training: {len(X_train_aug)} (with augmentation)
       • Validation: {len(X_val)}
       • Test: {len(X_test)}
    
    🎯 Model Performance:
       • Test Accuracy: {test_metrics['test_accuracy']*100:.2f}%
       • Test Loss: {test_metrics['test_loss']:.4f}
       • F1-Score (Macro): {f1_macro:.3f}
       • F1-Score (Weighted): {f1_weighted:.3f}
    
    📁 Generated Artifacts:
       ✓ models/best_model.h5 (Trained model)
       ✓ outputs/training_history.png (Learning curves)
       ✓ outputs/confusion_matrix.png (Classification matrix)
       ✓ outputs/sample_predictions.png (Visual examples)
       ✓ outputs/metrics.json (Detailed metrics)
       ✓ outputs/summary_report.md (Comprehensive report)
    
    💡 Recommendations:
       1. Review outputs/summary_report.md for full analysis
       2. Check outputs/confusion_matrix.png for misclassifications
       3. Examine outputs/sample_predictions.png for model behavior
    
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                    ✅ ALL TASKS COMPLETED SUCCESSFULLY                   ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Print improvement suggestions
    print_improvement_suggestions(test_metrics, CONFIG, gpu_available)
    
    return all_metrics, history


def print_improvement_suggestions(test_metrics, config, gpu_available):
    """Print suggestions for improving model performance"""
    
    print_section_header("💡 IMPROVEMENT SUGGESTIONS")
    
    accuracy = test_metrics['test_accuracy']
    
    print("\n🔧 Two Key Improvements for Better Classification:\n")
    
    print("1. DATA ENHANCEMENT:")
    print("   • Collect more real bite mark images (current: synthetic)")
    print("   • Increase dataset size to 500-1000 samples per class")
    print("   • Use data from forensic databases or medical literature")
    print("   • Apply more sophisticated augmentation (elastic deformation)")
    print("   • Consider ensemble methods with multiple models\n")
    
    print("2. MODEL ARCHITECTURE:")
    print("   • Fine-tune with pre-trained ImageNet weights (if using RGB)")
    print("   • Implement attention mechanisms to focus on bite patterns")
    print("   • Use ensemble of models (CNN + Vision Transformer)")
    print("   • Add spatial transformer networks for rotation invariance")
    print("   • Experiment with deeper architectures (ResNet, EfficientNet)\n")
    
    print("\n🚀 Recommended Lightweight Models for 4GB GPU:\n")
    
    print("1. MobileNetV3 (Recommended):")
    print("   • Parameters: ~4-5M (FP16: ~10MB)")
    print("   • Training speed: Fast")
    print("   • Accuracy: High")
    print("   • Memory: Low (~2GB VRAM)")
    print("   • Best for: Real-time inference\n")
    
    print("2. EfficientNet-B0:")
    print("   • Parameters: ~5.3M (FP16: ~11MB)")
    print("   • Training speed: Medium")
    print("   • Accuracy: Very High")
    print("   • Memory: Medium (~2.5GB VRAM)")
    print("   • Best for: Balanced accuracy/efficiency\n")
    
    print("3. ShuffleNetV2:")
    print("   • Parameters: ~2-3M (FP16: ~5MB)")
    print("   • Training speed: Very Fast")
    print("   • Accuracy: Good")
    print("   • Memory: Very Low (~1.5GB VRAM)")
    print("   • Best for: Extreme memory constraints\n")
    
    print("4. Custom Tiny CNN (Current):")
    print("   • Parameters: ~1-2M (FP16: ~3MB)")
    print("   • Training speed: Very Fast")
    print("   • Accuracy: Good for simple tasks")
    print("   • Memory: Minimal (~1GB VRAM)")
    print("   • Best for: Quick prototyping\n")
    
    print("💾 Current GPU Utilization:")
    print(f"   • GPU Available: {'Yes (4GB RTX)' if gpu_available else 'No (CPU mode)'}")
    print(f"   • Mixed Precision: {'Enabled (FP16)' if gpu_available else 'N/A'}")
    print(f"   • Batch Size: {config['batch_size']}")
    print(f"   • Estimated VRAM Usage: ~{'2-3GB' if gpu_available else 'N/A'}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        metrics, history = run_complete_pipeline()
        print("\n✅ Pipeline executed successfully!")
        print("📖 Check outputs/summary_report.md for detailed results\n")
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
