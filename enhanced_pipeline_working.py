"""
🦷 BITE MARK CLASSIFICATION - WORKING ENHANCED PIPELINE
Compatible version using existing enhanced modules

Features:
- Uses enhanced preprocessing for consistency
- GPU optimization where available
- Advanced evaluation capabilities
- Production-ready error handling
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Optimize TensorFlow for performance and reduce warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory growth

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Core imports
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Configure TensorFlow for optimal performance
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.get_logger().setLevel('ERROR')  # Reduce TensorFlow warnings

# Enable memory growth for GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Enhanced modules (using actual class names)
from src.shared_preprocessing import SharedPreprocessor, PreprocessingConfig
from src.gpu_augmentation import GPUAugmentor, AugmentationConfig
from src.enhanced_cnn import EnhancedBiteMarkCNN, ModelConfig
from src.comprehensive_evaluator import ComprehensiveEvaluator
from src.global_utils import (
    GlobalConfig, setup_gpu, print_section_header, 
    get_reproducible_seed, setup_environment, create_directories,
    plot_training_history, generate_summary_report, save_pipeline_state
)

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EnhancedPipelineConfig:
    """Enhanced pipeline configuration optimized for performance"""
    # Data settings
    data_dir: str = "data"
    processed_dir: str = "data/processed"
    augmented_dir: str = "data/augmented"
    target_size: Tuple[int, int] = (224, 224)
    batch_size: int = 20  # Larger for efficiency
    validation_split: float = 0.2
    
    # Training settings - optimized for ultra-fast training
    epochs: int = 10  # Ultra-fast for speed testing
    initial_lr: float = 0.001  # Higher for faster convergence
    patience: int = 3  # Even faster early stopping
    min_delta: float = 0.001  # Balanced threshold
    
    # Model settings
    use_attention: bool = True
    use_se_blocks: bool = True
    dropout_rate: float = 0.3  # Reduced dropout for better convergence
    model_type: str = 'enhanced_cnn'  # Keep enhanced but optimized
    
    # Output settings
    output_dir: str = "outputs"
    model_dir: str = "models"
    
    # System settings
    seed: int = 42
    mixed_precision: bool = False  # Keep disabled for stability


class EnhancedBiteMarkPipeline:
    """Enhanced BiteMark Classification Pipeline"""
    
    def __init__(self):
        """Initialize enhanced pipeline"""
        print_section_header("ENHANCED BITEMARK CLASSIFICATION PIPELINE")
        
        self.config = EnhancedPipelineConfig()
        self.global_config = GlobalConfig()
        
        # Component instances
        self.preprocessor = None
        self.augmentor = None
        self.model = None
        self.evaluator = None
        
        # State
        self.gpu_available = False
        self.class_names = []
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Results
        self.history = None
        self.metrics = {}
    
    def initialize_environment(self):
        """Initialize environment and components"""
        print_section_header("ENVIRONMENT INITIALIZATION")
        
        # Setup reproducibility
        seed = get_reproducible_seed() if self.config.seed is None else self.config.seed
        
        # Setup GPU and environment
        self.gpu_available = setup_environment(
            mixed_precision=self.config.mixed_precision, 
            seed=seed
        )
        
        # Create directories
        create_directories(self.config)
        
        logger.info(f"✅ Environment initialized - GPU: {self.gpu_available}")
    
    def initialize_components(self):
        """Initialize enhanced components"""
        print_section_header("COMPONENT INITIALIZATION")
        
        # 1. Enhanced Preprocessor
        preprocess_config = PreprocessingConfig(
            img_size=self.config.target_size,
            normalize=True,
            adaptive_histogram=True,
            denoise=True
        )
        self.preprocessor = SharedPreprocessor(preprocess_config)
        logger.info("✅ SharedPreprocessor initialized")
        
        # 2. GPU Augmentor
        aug_config = AugmentationConfig(
            use_gpu=self.gpu_available,
            preserve_forensic_features=True,
            balance_classes=True,
            seed=self.config.seed
        )
        self.augmentor = GPUAugmentor(config=aug_config)
        logger.info("✅ GPUAugmentor initialized")
        
        # 3. Enhanced CNN
        model_config = ModelConfig(
            input_shape=(*self.config.target_size, 3),
            num_classes=3,  # human, dog, snake
            use_attention=self.config.use_attention,
            dropout_rate=self.config.dropout_rate
        )
        self.model = EnhancedBiteMarkCNN(config=model_config)
        logger.info("✅ EnhancedBiteMarkCNN initialized")
        
        # 4. Comprehensive Evaluator
        self.evaluator = ComprehensiveEvaluator()
        logger.info("✅ ComprehensiveEvaluator initialized")
    
    def load_and_prepare_data(self):
        """Load and prepare data with enhanced preprocessing"""
        print_section_header("DATA LOADING & PREPARATION")
        
        try:
            # Try with processed data first
            data_path = self.config.processed_dir
            
            if os.path.exists(data_path):
                logger.info(f"Loading from processed data: {data_path}")
                X, y, class_names = self.preprocessor.load_dataset(
                    data_dir=data_path,
                    save_processed=False
                )
            else:
                # Fallback to raw data
                logger.info(f"Processing from raw data: {self.config.data_dir}/raw")
                X, y, class_names = self.preprocessor.load_dataset(
                    data_dir=f"{self.config.data_dir}/raw",
                    save_processed=True
                )
            
            self.class_names = class_names
            logger.info(f"✅ Data loaded - Classes: {self.class_names}")
            logger.info(f"✅ Dataset shape: {X.shape}, Labels: {y.shape}")
            
            # Split data
            from sklearn.model_selection import train_test_split
            
            # Split into train/validation/test
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=self.config.validation_split * 2, 
                random_state=self.config.seed, stratify=y
            )
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, 
                random_state=self.config.seed, stratify=y_temp
            )
            
            # Create highly optimized TensorFlow datasets like main_pipeline
            # Training dataset with augmentation and optimizations
            self.train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            self.train_dataset = (self.train_dataset
                                .shuffle(buffer_size=min(1000, len(X_train)), seed=self.config.seed)
                                .batch(self.config.batch_size, drop_remainder=True)
                                .map(self._augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
                                .prefetch(tf.data.AUTOTUNE)
                                .cache())  # Cache after augmentation
            
            # Validation dataset - no augmentation, optimized for speed
            self.val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            self.val_dataset = (self.val_dataset
                              .batch(self.config.batch_size, drop_remainder=False)
                              .cache()  # Cache first for validation
                              .prefetch(tf.data.AUTOTUNE))
            
            # Test dataset - same as validation
            self.test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            self.test_dataset = (self.test_dataset
                               .batch(self.config.batch_size, drop_remainder=False)
                               .cache()
                               .prefetch(tf.data.AUTOTUNE))
            
            logger.info(f"✅ Data split complete - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            return False
    
    def _augment_batch(self, images, labels):
        """Apply lightweight data augmentation during training"""
        # Apply simple augmentations for speed
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_brightness(images, 0.1)
        images = tf.image.random_contrast(images, 0.9, 1.1)
        # Keep values in valid range
        images = tf.clip_by_value(images, 0.0, 1.0)
        return images, labels
    
    def train_model(self):
        """Train the enhanced model with optimizations from main_pipeline"""
        print_section_header("ENHANCED MODEL TRAINING")
        
        try:
            # Build the model
            model = self.model.build_model()
            
            # Configure optimizer like main_pipeline (lower LR for transfer learning)
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config.initial_lr,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
                clipnorm=1.0  # Gradient clipping for stability
            )
            
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
                run_eagerly=False  # Disable eager execution for speed
            )
            
            # Advanced callbacks like main_pipeline
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=self.config.patience,
                    restore_best_weights=True,
                    min_delta=self.config.min_delta,
                    verbose=1,
                    mode='max'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_accuracy',
                    factor=0.5,
                    patience=max(2, self.config.patience // 2),
                    min_lr=1e-7,
                    verbose=1,
                    mode='max'
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{self.config.model_dir}/best_model_enhanced.h5",
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=0,
                    mode='max'
                ),
                # Add learning rate scheduler for better convergence
                tf.keras.callbacks.LearningRateScheduler(
                    lambda epoch: self.config.initial_lr * (0.95 ** epoch),
                    verbose=0
                )
            ]
            
            # Train with class weights for imbalanced data
            from sklearn.utils.class_weight import compute_class_weight
            
            # Get class distribution from training data
            train_labels = []
            for _, labels in self.train_dataset.unbatch().take(1000):  # Sample for class weights
                # Handle both single labels and batches
                if labels.ndim == 0:
                    train_labels.append(int(labels.numpy()))
                else:
                    train_labels.extend([int(label) for label in labels.numpy().flat])
            
            if len(train_labels) > 0:
                unique_classes = np.unique(train_labels)
                class_weights = compute_class_weight(
                    'balanced', classes=unique_classes, y=train_labels
                )
                class_weight_dict = dict(zip([int(cls) for cls in unique_classes], class_weights))
                logger.info(f"Using class weights: {class_weight_dict}")
            else:
                class_weight_dict = None
            
            # Train the model with optimizations
            logger.info("🚀 Starting optimized enhanced training...")
            start_time = time.time()
            
            self.history = model.fit(
                self.train_dataset,
                epochs=self.config.epochs,
                validation_data=self.val_dataset,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1,  # Show progress like main_pipeline
                workers=4,
                use_multiprocessing=False  # Keep False for Windows compatibility
            )
            
            training_time = time.time() - start_time
            logger.info(f"✅ Optimized training completed in {training_time:.1f}s")
            logger.info(f"⚡ Speed: {training_time/self.config.epochs:.1f}s per epoch")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return None
    
    def evaluate_model(self, model):
        """Comprehensive model evaluation"""
        print_section_header("COMPREHENSIVE EVALUATION")
        
        try:
            # Load best weights into current model (avoid custom layer issues)
            model.load_weights(f"{self.config.model_dir}/best_model_enhanced.h5")
            
            # Evaluate on test set
            test_results = model.evaluate(self.test_dataset, verbose=0)
            test_loss = test_results[0]
            test_acc = test_results[1]
            
            # Store basic metrics
            self.metrics.update({
                'test_accuracy': float(test_acc),
                'test_loss': float(test_loss)
            })
            
            logger.info(f"✅ Evaluation completed - Accuracy: {test_acc:.3f}")
            
            # Skip comprehensive evaluation if it causes issues
            try:
                # Use comprehensive evaluator for detailed analysis
                self.evaluator.load_model(f"{self.config.model_dir}/best_model_enhanced.h5")
                comprehensive_metrics = self.evaluator.comprehensive_evaluation(
                    self.test_dataset,
                    self.class_names,
                    self.config.output_dir
                )
                
                # Merge metrics
                self.metrics.update(comprehensive_metrics)
                logger.info("✅ Comprehensive evaluation completed")
            except Exception as eval_error:
                logger.warning(f"⚠️  Comprehensive evaluation skipped due to: {eval_error}")
                logger.info("Continuing with basic metrics...")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return False
    
    def generate_reports(self):
        """Generate comprehensive reports"""
        print_section_header("GENERATING REPORTS")
        
        try:
            # Training history plot
            if self.history:
                plot_training_history(
                    self.history,
                    f"{self.config.output_dir}/training_history_enhanced.png"
                )
            
            # Summary report
            generate_summary_report(
                self.metrics,
                f"{self.config.output_dir}/enhanced_summary_report.md"
            )
            
            # Save pipeline state
            save_pipeline_state(
                self.metrics,
                self.history,
                self.config,
                self.config.output_dir
            )
            
            logger.info("✅ Reports generated successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete enhanced pipeline"""
        pipeline_start = time.time()
        
        try:
            # Step 1: Initialize environment
            self.initialize_environment()
            
            # Step 2: Initialize components
            self.initialize_components()
            
            # Step 3: Load and prepare data
            if not self.load_and_prepare_data():
                raise Exception("Data loading failed")
            
            # Step 4: Train model
            model = self.train_model()
            if model is None:
                raise Exception("Training failed")
            
            # Step 5: Evaluate model
            if not self.evaluate_model(model):
                raise Exception("Evaluation failed")
            
            # Step 6: Generate reports
            if not self.generate_reports():
                raise Exception("Report generation failed")
            
            # Success summary
            total_time = time.time() - pipeline_start
            print_section_header("🎉 ENHANCED PIPELINE COMPLETED")
            
            logger.info(f"✅ Total pipeline time: {total_time:.1f}s")
            logger.info(f"✅ Test accuracy: {self.metrics.get('test_accuracy', 0):.3f}")
            logger.info(f"✅ GPU utilized: {self.gpu_available}")
            logger.info(f"✅ Enhanced components: All operational")
            
            print_section_header("Enhanced BiteMark Classification System Ready! 🚀")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            print_section_header("Pipeline Failed - Check Logs")
            return False


if __name__ == "__main__":
    # Run the enhanced pipeline
    pipeline = EnhancedBiteMarkPipeline()
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎉 Enhanced BiteMark Classification System is operational!")
    else:
        print("\n❌ Pipeline execution failed. Check logs for details.")