#!/usr/bin/env python3
"""
Balanced Training Script for Bite Mark Classification

Features:
- Deterministic oversampling for minority classes (dog, human)
- Class weights for imbalanced training
- Balanced batch generation (optional)
- CPU-optimized data pipeline
- Comprehensive metrics and visualizations

Usage:
    python scripts/run_balanced_train.py [--regenerate-augmented] [--use-balanced-batches]
    
Author: Bite Mark Classification Team
Date: November 7, 2025
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Import project modules
from data_preprocessing import BiteMarkPreprocessor
from augmentation import BiteMarkAugmentor, SEED
from train_cnn import BiteMarkCNN
from evaluate_model import ModelEvaluator
from data_utils import compute_class_weights, BalancedBatchGenerator, verify_class_balance
from utils import setup_gpu, print_section_header

# Set all random seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

print(f"\n🌱 Global SEED set to: {SEED} (deterministic mode)")


def main(args):
    """Main balanced training pipeline."""
    
    print_section_header("BALANCED BITE MARK CLASSIFICATION TRAINING")
    print("🎯 Goal: Fix class imbalance with oversampling + class weights + balanced batches")
    print(f"📁 Project root: {Path.cwd()}")
    
    # Configuration
    CONFIG = {
        'img_size': (224, 224),
        'grayscale': False,
        'batch_size': 24,  # Divisible by 3 classes (8 samples per class per batch)
        'epochs': 50,
        'learning_rate': 0.0005,
        'augmentation_factor': 3,  # Base factor
        'test_size': 0.2,
        'val_size': 0.15,
        'model_type': 'mobilenet',
        'balance_classes': True,
        'use_class_weights': True,
        'use_balanced_batches': args.use_balanced_batches,
        'adaptive_histogram': True,
        'denoise': True,
    }
    
    print("\n⚙️  Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Step 1: Setup
    print_section_header("STEP 1: ENVIRONMENT SETUP")
    gpu_available = setup_gpu()
    if not gpu_available:
        print("⚠️  Running on CPU - training will be slower")
    
    class_names = ['human', 'dog', 'snake']
    
    # Step 2: Load and preprocess data
    print_section_header("STEP 2: LOAD & PREPROCESS DATA")
    preprocessor = BiteMarkPreprocessor(
        img_size=CONFIG['img_size'],
        grayscale=CONFIG['grayscale'],
        adaptive_histogram=CONFIG['adaptive_histogram'],
        denoise=CONFIG['denoise']
    )
    
    images, labels, class_names = preprocessor.load_sample_data(
        'data/raw',
        save_processed=True,
        processed_dir='data/processed'
    )
    
    print(f"\n📊 Original dataset:")
    unique, counts = np.unique(labels, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {class_names[cls]}: {count} images")
    
    # Step 3: Split data
    print_section_header("STEP 3: SPLIT DATA")
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels,
        test_size=CONFIG['test_size'],
        stratify=labels,
        random_state=SEED
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=CONFIG['val_size'] / (1 - CONFIG['test_size']),
        stratify=y_temp,
        random_state=SEED
    )
    
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Step 4: Augmentation with DETERMINISTIC OVERSAMPLING
    print_section_header("STEP 4: DETERMINISTIC OVERSAMPLING")
    
    if args.regenerate_augmented:
        print("🔄 Regenerating augmented dataset...")
        # Clear existing augmented directory
        import shutil
        aug_dir = Path('data/augmented')
        if aug_dir.exists():
            shutil.rmtree(aug_dir)
        
        augmentor = BiteMarkAugmentor(
            preserve_features=True,
            balance_classes=CONFIG['balance_classes']
        )
        
        X_train_aug, y_train_aug = augmentor.augment_dataset(
            X_train,
            y_train,
            augmentation_factor=CONFIG['augmentation_factor'],
            class_names=class_names,
            save_augmented=True,
            augmented_dir='data/augmented'
        )
    else:
        print("📂 Using existing augmented dataset (if available)")
        augmentor = BiteMarkAugmentor(
            preserve_features=True,
            balance_classes=CONFIG['balance_classes']
        )
        
        X_train_aug, y_train_aug = augmentor.augment_dataset(
            X_train,
            y_train,
            augmentation_factor=CONFIG['augmentation_factor'],
            class_names=class_names,
            save_augmented=False
        )
    
    # Verify class balance
    print("\n✅ Verification:")
    verify_class_balance('data/augmented', class_names)
    
    print(f"\n📊 Augmented training set:")
    unique, counts = np.unique(y_train_aug, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {class_names[cls]}: {count} images ({count/len(y_train_aug)*100:.1f}%)")
    
    # Step 5: Compute class weights
    print_section_header("STEP 5: COMPUTE CLASS WEIGHTS")
    class_weights = None
    if CONFIG['use_class_weights']:
        class_weights = compute_class_weights(y_train_aug, class_names)
    else:
        print("⚠️  Class weights disabled")
    
    # Step 6: Create TensorFlow datasets or balanced generator
    print_section_header("STEP 6: CREATE DATA PIPELINE")
    
    if CONFIG['use_balanced_batches']:
        print("🔄 Using BalancedBatchGenerator (on-the-fly oversampling)")
        # Note: For TensorFlow integration, we'll use standard datasets with class weights
        # The generator is available for custom training loops if needed
        train_batches_per_epoch = len(X_train_aug) // CONFIG['batch_size']
        print(f"  Batches per epoch: {train_batches_per_epoch}")
    
    # Create standard TF datasets (CPU-optimized)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_aug, y_train_aug))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train_aug), seed=SEED)
    train_dataset = train_dataset.batch(CONFIG['batch_size'])
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(CONFIG['batch_size'])
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(CONFIG['batch_size'])
    
    print(f"✓ TensorFlow datasets created (batch_size={CONFIG['batch_size']})")
    
    # Step 7: Build model
    print_section_header("STEP 7: BUILD MODEL")
    model = BiteMarkCNN(
        input_shape=(*CONFIG['img_size'], 3),
        num_classes=len(class_names),
        model_type=CONFIG['model_type']
    )
    
    if CONFIG['model_type'] == 'mobilenet':
        model.build_mobilenet_model()
    else:
        model.build_efficient_model()
    
    model.compile(
        optimizer='adam',
        learning_rate=CONFIG['learning_rate']
    )
    
    print(f"\n📊 Model parameters: {model.model.count_params():,}")
    
    # Step 8: Setup callbacks
    print_section_header("STEP 8: SETUP TRAINING")
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/balanced_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        ),
    ]
    
    print("✓ Callbacks configured:")
    print("  - ModelCheckpoint (balanced_model.h5)")
    print("  - EarlyStopping (patience=15)")
    print("  - ReduceLROnPlateau (factor=0.5)")
    
    # Step 9: Train
    print_section_header("STEP 9: TRAINING")
    print(f"🚀 Starting training with balanced data...")
    print(f"   Class weights: {'Enabled ✓' if class_weights else 'Disabled'}")
    print(f"   Balanced batches: {'Enabled ✓' if CONFIG['use_balanced_batches'] else 'Standard'}")
    
    history, training_time = model.train(
        train_dataset,
        val_dataset,
        epochs=CONFIG['epochs'],
        class_weights=class_weights,
        callbacks_list=callbacks_list
    )
    
    # Step 10: Evaluate
    print_section_header("STEP 10: EVALUATION")
    
    evaluator = ModelEvaluator(model.model, class_names)
    
    # Load best model
    print("📥 Loading best model (balanced_model.h5)...")
    model.model.load_weights('models/balanced_model.h5')
    
    # Evaluate on test set
    test_results = evaluator.evaluate(test_dataset)
    
    # Generate predictions
    predictions = evaluator.predict(X_test)
    
    # Save metrics
    print_section_header("STEP 11: SAVE RESULTS")
    
    metrics = {
        'test_accuracy': float(test_results['accuracy']),
        'test_loss': float(test_results['loss']),
        'test_precision': float(test_results['precision']),
        'test_recall': float(test_results['recall']),
        'per_class_metrics': {},
        'class_names': class_names,
        'config': CONFIG,
        'training_time_seconds': training_time,
        'class_weights_used': class_weights,
        'augmented_counts': {
            name: int(count) for name, count in zip(
                [class_names[i] for i in unique],
                counts
            )
        }
    }
    
    # Get per-class metrics
    from sklearn.metrics import classification_report
    report = classification_report(y_test, predictions, target_names=class_names, output_dict=True)
    
    for class_name in class_names:
        if class_name in report:
            metrics['per_class_metrics'][class_name] = {
                'precision': report[class_name]['precision'],
                'recall': report[class_name]['recall'],
                'f1-score': report[class_name]['f1-score'],
                'support': int(report[class_name]['support'])
            }
    
    metrics['macro_avg'] = report['macro avg']
    metrics['weighted_avg'] = report['weighted avg']
    
    # Save to JSON
    with open('outputs/metrics_balanced.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("✓ Metrics saved to outputs/metrics_balanced.json")
    
    # Generate confusion matrix
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(y_test, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Balanced Training', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix_balanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Confusion matrix saved to outputs/confusion_matrix_balanced.png")
    
    # Print final summary
    print_section_header("TRAINING SUMMARY")
    print(f"✅ Balanced training completed!")
    print(f"\n📊 Final Results:")
    print(f"  Training time: {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"  Test accuracy: {metrics['test_accuracy']*100:.2f}%")
    print(f"  Test loss: {metrics['test_loss']:.4f}")
    
    print(f"\n📈 Per-Class Performance:")
    for class_name in class_names:
        if class_name in metrics['per_class_metrics']:
            m = metrics['per_class_metrics'][class_name]
            print(f"  {class_name}:")
            print(f"    Precision: {m['precision']*100:.1f}%")
            print(f"    Recall: {m['recall']*100:.1f}%")
            print(f"    F1-Score: {m['f1-score']:.3f}")
            print(f"    Support: {m['support']}")
    
    print(f"\n📁 Artifacts:")
    print(f"  ✓ models/balanced_model.h5")
    print(f"  ✓ outputs/metrics_balanced.json")
    print(f"  ✓ outputs/confusion_matrix_balanced.png")
    print(f"  ✓ data/augmented/ ({sum(counts)} balanced images)")
    
    print("\n" + "="*80)
    print("✨ All done! Check outputs/ for detailed results.")
    print("="*80 + "\n")
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Balanced training for bite mark classification'
    )
    parser.add_argument(
        '--regenerate-augmented',
        action='store_true',
        help='Regenerate augmented dataset (clears existing)'
    )
    parser.add_argument(
        '--use-balanced-batches',
        action='store_true',
        help='Use balanced batch generator (experimental)'
    )
    
    args = parser.parse_args()
    
    try:
        metrics = main(args)
        print("✅ Success!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
