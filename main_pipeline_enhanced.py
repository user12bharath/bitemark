"""
🦷 BITE MARK CLASSIFICATION - ENHANCED COMPLETE PIPELINE
Production-grade end-to-end training and evaluation system
Optimized for modern GPU architectures with advanced ML techniques

Features:
- Shared preprocessing for train-inference consistency
- GPU-optimized augmentation pipeline
- Advanced CNN with attention mechanisms
- Comprehensive evaluation with ROC/AUC analysis
- Production-ready configuration management
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Core imports
import numpy as np
import tensorflow as tf

# Custom modules
from src.shared_preprocessing import SharedPreprocessor, PreprocessingConfig
from src.gpu_augmentation import GPUAugmentor, AugmentationConfig
from src.enhanced_cnn import EnhancedBiteMarkCNN, ModelConfig
from src.comprehensive_evaluator import ComprehensiveEvaluator
from src.global_utils import (
    GlobalConfig, setup_gpu, print_section_header, 
    get_reproducible_seed, setup_environment, create_directories,
    plot_training_history, generate_summary_report, save_pipeline_state
)


@dataclass
class PipelineConfig:
    """Centralized pipeline configuration"""
    # Data parameters
    img_size: Tuple[int, int] = (224, 224)
    channels: int = 3  # RGB for real forensic images
    
    # Training parameters
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 0.0005
    
    # Data split parameters
    test_size: float = 0.2
    val_size: float = 0.15
    
    # Augmentation parameters
    augmentation_factor: int = 3
    balance_classes: bool = True
    
    # Model parameters
    model_type: str = 'enhanced_cnn'  # 'enhanced_cnn', 'mobilenet', 'efficientnet'
    use_attention: bool = True
    use_mixed_precision: bool = True
    
    # Transfer learning schedule
    freeze_epochs: int = 10
    warmup_epochs: int = 5
    unfreeze_epochs: int = 25
    
    # Paths
    data_dir: str = 'data/raw'
    processed_dir: str = 'data/processed'
    augmented_dir: str = 'data/augmented'
    model_dir: str = 'models'
    output_dir: str = 'outputs'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'img_size': self.img_size,
            'channels': self.channels,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'test_size': self.test_size,
            'val_size': self.val_size,
            'augmentation_factor': self.augmentation_factor,
            'balance_classes': self.balance_classes,
            'model_type': self.model_type,
            'use_attention': self.use_attention,
            'use_mixed_precision': self.use_mixed_precision,
            'freeze_epochs': self.freeze_epochs,
            'warmup_epochs': self.warmup_epochs,
            'unfreeze_epochs': self.unfreeze_epochs
        }


class EnhancedPipeline:
    """Production-grade BiteMark classification pipeline"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize pipeline with configuration"""
        self.config = config or PipelineConfig()
        self.global_config = GlobalConfig()
        self.pipeline_start_time = None
        self.gpu_available = False
        
        # Initialize components
        self.preprocessor: Optional[SharedPreprocessor] = None
        self.augmentor: Optional[GPUAugmentor] = None
        self.model: Optional[EnhancedBiteMarkCNN] = None
        self.evaluator: Optional[ComprehensiveEvaluator] = None
        
        # Data storage
        self.class_names = []
        self.training_history = None
        self.final_metrics = {}
    
    def initialize_environment(self) -> bool:
        """Setup environment and hardware"""
        print_section_header("INITIALIZATION & ENVIRONMENT SETUP")
        
        # Create directories
        create_directories(self.config)
        
        # Setup reproducible environment
        seed = get_reproducible_seed()
        self.gpu_available = setup_environment(
            mixed_precision=self.config.use_mixed_precision,
            seed=seed
        )
        
        # Adjust batch size based on GPU memory
        if self.gpu_available:
            self.config.batch_size = min(32, self.config.batch_size)
        else:
            self.config.batch_size = max(8, self.config.batch_size // 2)
        
        print(f"\n⚙️  Pipeline Configuration:")
        for key, value in self.config.to_dict().items():
            print(f"  {key:.<25} {value}")
        
        return self.gpu_available
    
    def initialize_components(self):
        """Initialize all pipeline components"""
        print_section_header("COMPONENT INITIALIZATION")
        
        # Shared preprocessing configuration
        prep_config = PreprocessingConfig(
            img_size=self.config.img_size,
            channels=self.config.channels,
            normalize=True,
            adaptive_histogram=True,
            denoise=True,
            clahe_clip_limit=2.0
        )
        
        # Initialize shared preprocessor
        self.preprocessor = SharedPreprocessor(config=prep_config)
        print("✓ Shared preprocessor initialized")
        
        # GPU-optimized augmentation configuration
        aug_config = AugmentationConfig(
            use_gpu=self.gpu_available,
            preserve_forensic_features=True,
            balance_classes=self.config.balance_classes
        )
        
        # Initialize GPU augmentor
        self.augmentor = GPUAugmentor(config=aug_config)
        print("✓ GPU augmentor initialized")
        
        # Model configuration
        input_shape = (*self.config.img_size, self.config.channels)
        model_config = ModelConfig(
            input_shape=input_shape,
            model_type=self.config.model_type,
            use_attention=self.config.use_attention,
            use_mixed_precision=self.config.use_mixed_precision,
            learning_rate=self.config.learning_rate
        )
        
        # Initialize enhanced CNN
        self.model = EnhancedBiteMarkCNN(config=model_config)
        print("✓ Enhanced CNN initialized")
        
        # Initialize comprehensive evaluator
        self.evaluator = ComprehensiveEvaluator()
        print("✓ Comprehensive evaluator initialized")
    
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                           np.ndarray, np.ndarray, np.ndarray]:
        """Load and prepare training data"""
        print_section_header("DATA LOADING & PREPARATION")
        
        # Load raw data using shared preprocessor
        print("📂 Loading dataset...")
        images, labels, self.class_names = self.preprocessor.load_dataset(
            data_dir=self.config.data_dir,
            save_processed=True,
            processed_dir=self.config.processed_dir
        )
        
        print(f"✓ Dataset loaded: {len(images)} samples")
        print(f"  Classes: {self.class_names}")
        print(f"  Image shape: {images[0].shape}")
        print(f"  Label distribution: {dict(zip(self.class_names, np.bincount(labels)))}")
        
        # Split data with stratification
        print("\n📊 Splitting dataset...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(
            images, labels, 
            test_size=self.config.test_size, 
            val_size=self.config.val_size,
            random_state=self.global_config.SEED
        )
        
        print(f"  Training: {len(X_train)} samples ({len(X_train)/len(images)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(images)*100:.1f}%)")
        print(f"  Test: {len(X_test)} samples ({len(X_test)/len(images)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def apply_augmentation(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply GPU-optimized augmentation with class balancing"""
        print_section_header("GPU-OPTIMIZED DATA AUGMENTATION")
        
        print(f"🔄 Applying {self.config.augmentation_factor}x augmentation with class balancing...")
        
        # Apply GPU-accelerated augmentation
        X_train_aug, y_train_aug = self.augmentor.augment_dataset(
            X_train, y_train,
            augmentation_factor=self.config.augmentation_factor,
            class_names=self.class_names,
            save_augmented=True,
            augmented_dir=self.config.augmented_dir
        )
        
        print(f"✓ Augmentation complete:")
        print(f"  Original: {len(X_train)} → Augmented: {len(X_train_aug)} samples")
        
        # Print final class distribution
        unique_labels, counts = np.unique(y_train_aug, return_counts=True)
        print(f"\n  Final distribution:")
        for label, count in zip(unique_labels, counts):
            class_name = self.class_names[label]
            percentage = (count / len(y_train_aug)) * 100
            print(f"    {class_name}: {count} samples ({percentage:.1f}%)")
        
        return X_train_aug, y_train_aug
    
    def create_datasets(self, X_train_aug: np.ndarray, y_train_aug: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Create optimized TensorFlow datasets"""
        print_section_header("TENSORFLOW DATASET OPTIMIZATION")
        
        # Create datasets with advanced optimizations
        train_dataset = self.preprocessor.create_optimized_dataset(
            X_train_aug, y_train_aug,
            batch_size=self.config.batch_size,
            shuffle=True,
            augment=True,
            cache=True,
            prefetch=True
        )
        
        val_dataset = self.preprocessor.create_optimized_dataset(
            X_val, y_val,
            batch_size=self.config.batch_size,
            shuffle=False,
            augment=False,
            cache=True,
            prefetch=True
        )
        
        test_dataset = self.preprocessor.create_optimized_dataset(
            X_test, y_test,
            batch_size=self.config.batch_size,
            shuffle=False,
            augment=False,
            cache=False,
            prefetch=True
        )
        
        print("✓ Optimized TensorFlow datasets created")
        print(f"  Training batches: ~{len(X_train_aug) // self.config.batch_size}")
        print(f"  Validation batches: ~{len(X_val) // self.config.batch_size}")
        print(f"  Test batches: ~{len(X_test) // self.config.batch_size}")
        print(f"  Optimizations: Caching, Prefetching, AUTOTUNE")
        
        return train_dataset, val_dataset, test_dataset
    
    def train_model(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset,
                   y_train_aug: np.ndarray) -> Dict[str, Any]:
        """Train model with advanced techniques"""
        print_section_header("ADVANCED MODEL TRAINING")
        
        # Calculate class weights for imbalanced data
        class_weights = self.model.calculate_class_weights(y_train_aug)
        
        # Build and compile model
        model = self.model.build_model(num_classes=len(self.class_names))
        self.model.compile_model()
        
        # Print model summary
        print("\n📐 Model Architecture:")
        model.summary()
        print(f"\n✓ Model parameters: {model.count_params():,}")
        
        # Create callbacks with advanced scheduling
        callbacks = self.model.create_advanced_callbacks(
            model_dir=self.config.model_dir,
            output_dir=self.config.output_dir,
            monitor_metric='val_accuracy'
        )
        
        print(f"\n🚀 Starting multi-phase training...")
        print("="*80)
        
        # Phase 1: Frozen backbone (if transfer learning)
        if self.config.model_type in ['mobilenet', 'efficientnet'] and self.config.freeze_epochs > 0:
            print(f"\n📚 Phase 1: Frozen backbone training ({self.config.freeze_epochs} epochs)")
            self.model.freeze_backbone()
            
            history_frozen = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.config.freeze_epochs,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
        
        # Phase 2: Warmup with low learning rate
        if self.config.warmup_epochs > 0:
            print(f"\n🔥 Phase 2: Warmup training ({self.config.warmup_epochs} epochs)")
            self.model.set_learning_rate(self.config.learning_rate * 0.1)
            
            history_warmup = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.config.warmup_epochs,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1,
                initial_epoch=self.config.freeze_epochs if self.config.freeze_epochs > 0 else 0
            )
        
        # Phase 3: Full training (unfreeze and fine-tune)
        print(f"\n🎯 Phase 3: Full training ({self.config.unfreeze_epochs} epochs)")
        self.model.unfreeze_backbone()
        self.model.set_learning_rate(self.config.learning_rate)
        
        initial_epoch = (self.config.freeze_epochs if self.config.freeze_epochs > 0 else 0) + \
                       (self.config.warmup_epochs if self.config.warmup_epochs > 0 else 0)
        
        self.training_history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
            initial_epoch=initial_epoch
        )
        
        print("="*80)
        print("✓ Multi-phase training completed!")
        
        training_metrics = {
            'total_epochs': len(self.training_history.history['loss']),
            'best_val_accuracy': max(self.training_history.history['val_accuracy']),
            'final_train_accuracy': self.training_history.history['accuracy'][-1],
            'final_val_accuracy': self.training_history.history['val_accuracy'][-1],
            'class_weights_used': True,
            'mixed_precision_used': self.config.use_mixed_precision
        }
        
        return training_metrics
    
    def evaluate_model(self, test_dataset: tf.data.Dataset) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        print_section_header("COMPREHENSIVE MODEL EVALUATION")
        
        # Load best model
        model_path = os.path.join(self.config.model_dir, 'best_model.h5')
        self.evaluator.load_model(model_path)
        
        # Comprehensive evaluation
        evaluation_results = self.evaluator.comprehensive_evaluation(
            test_dataset=test_dataset,
            class_names=self.class_names,
            output_dir=self.config.output_dir
        )
        
        print("✓ Comprehensive evaluation completed")
        print(f"  Test Accuracy: {evaluation_results['test_accuracy']*100:.2f}%")
        print(f"  Macro F1-Score: {evaluation_results['f1_macro']:.3f}")
        print(f"  Weighted F1-Score: {evaluation_results['f1_weighted']:.3f}")
        print(f"  Macro AUC: {evaluation_results['auc_macro']:.3f}")
        
        return evaluation_results
    
    def generate_visualizations(self):
        """Generate all visualizations and reports"""
        print_section_header("VISUALIZATION & REPORTING")
        
        # Generate training visualizations
        if self.training_history:
            self.evaluator.plot_training_history(
                self.training_history,
                output_path=os.path.join(self.config.output_dir, 'training_history.png')
            )
        
        print("✓ All visualizations generated")
        print(f"  Training history: {self.config.output_dir}/training_history.png")
        print(f"  Confusion matrix: {self.config.output_dir}/confusion_matrix.png")
        print(f"  ROC curves: {self.config.output_dir}/roc_curves.png")
        print(f"  Sample predictions: {self.config.output_dir}/sample_predictions.png")
        print(f"  Misclassified samples: {self.config.output_dir}/misclassified_samples.png")
        print(f"  Calibration curves: {self.config.output_dir}/calibration_curves.png")
    
    def save_pipeline_artifacts(self, training_metrics: Dict[str, Any], 
                               evaluation_results: Dict[str, Any]):
        """Save all pipeline artifacts and reports"""
        print_section_header("SAVING ARTIFACTS & REPORTS")
        
        # Compile comprehensive metrics
        self.final_metrics = {
            **evaluation_results,
            **training_metrics,
            'pipeline_config': self.config.to_dict(),
            'total_pipeline_time': time.time() - self.pipeline_start_time,
            'gpu_used': self.gpu_available,
            'reproducible_seed': self.global_config.SEED
        }
        
        # Save pipeline state
        save_pipeline_state(
            metrics=self.final_metrics,
            history=self.training_history,
            config=self.config,
            output_dir=self.config.output_dir
        )
        
        print("✓ Pipeline artifacts saved:")
        print(f"  Metrics: {self.config.output_dir}/metrics.json")
        print(f"  Config: {self.config.output_dir}/pipeline_config.json")
        print(f"  Summary: {self.config.output_dir}/summary_report.md")
        print(f"  Model: {self.config.model_dir}/best_model.h5")
    
    def print_final_summary(self):
        """Print comprehensive final summary"""
        pipeline_time = time.time() - self.pipeline_start_time
        
        print_section_header("🎉 PIPELINE EXECUTION COMPLETE")
        
        print(f"""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                         EXECUTION SUMMARY                                ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    
    ⏱️  Total Pipeline Time: {pipeline_time:.2f}s ({pipeline_time/60:.2f} minutes)
    
    🎯 Final Performance:
       • Test Accuracy: {self.final_metrics.get('test_accuracy', 0)*100:.2f}%
       • Macro F1-Score: {self.final_metrics.get('f1_macro', 0):.3f}
       • Weighted F1-Score: {self.final_metrics.get('f1_weighted', 0):.3f}
       • Macro AUC: {self.final_metrics.get('auc_macro', 0):.3f}
    
    📊 Training Details:
       • Total Epochs: {self.final_metrics.get('total_epochs', 0)}
       • Best Val Accuracy: {self.final_metrics.get('best_val_accuracy', 0)*100:.2f}%
       • GPU Acceleration: {'✓' if self.gpu_available else '✗'}
       • Mixed Precision: {'✓' if self.config.use_mixed_precision else '✗'}
    
    📁 Generated Artifacts:
       ✓ models/best_model.h5 (Production model)
       ✓ outputs/comprehensive_metrics.json (All metrics)
       ✓ outputs/training_history.png (Learning curves)
       ✓ outputs/roc_curves.png (ROC/AUC analysis)
       ✓ outputs/confusion_matrix.png (Classification matrix)
       ✓ outputs/sample_predictions.png (Visual examples)
       ✓ outputs/misclassified_samples.png (Error analysis)
       ✓ outputs/calibration_curves.png (Confidence calibration)
       ✓ outputs/summary_report.md (Comprehensive report)
    
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              ✅ PRODUCTION-READY MODEL SUCCESSFULLY CREATED              ║
    ╚══════════════════════════════════════════════════════════════════════════╝
        """)
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete enhanced pipeline"""
        self.pipeline_start_time = time.time()
        
        try:
            # ASCII Art Header
            print("\n" + "="*80)
            print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║           🦷  ENHANCED BITE MARK CLASSIFICATION SYSTEM  🦷               ║
    ║                                                                          ║
    ║         Production-Grade Deep Learning Pipeline for Forensics            ║
    ║              Advanced CNN • GPU Optimization • ROC Analysis              ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
            """)
            print("="*80)
            
            # Phase 1: Initialize environment and components
            self.initialize_environment()
            self.initialize_components()
            
            # Phase 2: Load and prepare data
            X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_prepare_data()
            
            # Phase 3: Apply augmentation
            X_train_aug, y_train_aug = self.apply_augmentation(X_train, y_train)
            
            # Phase 4: Create optimized datasets
            train_dataset, val_dataset, test_dataset = self.create_datasets(
                X_train_aug, y_train_aug, X_val, y_val, X_test, y_test
            )
            
            # Phase 5: Train model
            training_metrics = self.train_model(train_dataset, val_dataset, y_train_aug)
            
            # Phase 6: Comprehensive evaluation
            evaluation_results = self.evaluate_model(test_dataset)
            
            # Phase 7: Generate visualizations
            self.generate_visualizations()
            
            # Phase 8: Save artifacts and reports
            self.save_pipeline_artifacts(training_metrics, evaluation_results)
            
            # Phase 9: Print final summary
            self.print_final_summary()
            
            return self.final_metrics
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            # Cleanup GPU memory
            if self.gpu_available:
                tf.keras.backend.clear_session()


def main():
    """Main entry point"""
    try:
        # Create enhanced pipeline with configuration
        config = PipelineConfig(
            img_size=(224, 224),
            channels=3,  # RGB for real forensic images
            batch_size=16,
            epochs=50,  # Reduced for demo, increase for production
            learning_rate=0.0005,
            augmentation_factor=3,
            use_attention=True,
            use_mixed_precision=True,
            model_type='enhanced_cnn'
        )
        
        pipeline = EnhancedPipeline(config)
        metrics = pipeline.run_complete_pipeline()
        
        print("\n✅ Enhanced pipeline executed successfully!")
        print("📖 Check outputs/summary_report.md for detailed analysis")
        
        return metrics
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return None
    
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {str(e)}")
        return None


if __name__ == "__main__":
    main()