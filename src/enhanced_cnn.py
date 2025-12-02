"""
🧠 ENHANCED CNN WITH ATTENTION MECHANISMS
Advanced deep learning architecture for forensic bite mark classification

Features:
- Attention mechanisms for focus on relevant features
- Efficient architecture optimized for 4GB GPU
- Transfer learning with progressive unfreezing
- Advanced regularization and optimization
- Multi-scale feature extraction
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV3Small, EfficientNetB0
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for enhanced CNN model - optimized for speed"""
    # Input parameters
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    num_classes: int = 3
    
    # Model architecture - optimized for fast training
    model_type: str = 'enhanced_cnn'  # 'enhanced_cnn', 'mobilenet', 'efficientnet'
    use_attention: bool = True
    use_mixed_precision: bool = False  # Keep disabled for stability
    
    # Training parameters - optimized
    learning_rate: float = 0.0005  # Lower like main_pipeline
    optimizer: str = 'adam'  # Simplified
    
    # Regularization - lighter
    dropout_rate: float = 0.3  # Reduced for faster convergence
    l2_regularization: float = 1e-5  # Reduced regularization
    
    # Architecture specifics - lighter
    base_filters: int = 24  # Reduced from 32
    growth_rate: float = 1.3  # Reduced growth
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'model_type': self.model_type,
            'use_attention': self.use_attention,
            'use_mixed_precision': self.use_mixed_precision,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
            'dropout_rate': self.dropout_rate,
            'l2_regularization': self.l2_regularization,
            'base_filters': self.base_filters,
            'growth_rate': self.growth_rate
        }


class AttentionModule(layers.Layer):
    """Lightweight attention module for focusing on relevant features"""
    
    def __init__(self, filters: int, name: str = 'attention', **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        
        # Simplified attention - just use global average pooling and channel attention
        self.global_avg_pool = layers.GlobalAveragePooling2D(name=f'{name}_gap')
        self.fc1 = layers.Dense(filters // 4, activation='relu', name=f'{name}_fc1')
        self.fc2 = layers.Dense(filters, activation='sigmoid', name=f'{name}_fc2')
        self.reshape = layers.Reshape((1, 1, filters), name=f'{name}_reshape')
        self.multiply = layers.Multiply(name=f'{name}_multiply')
        
    def call(self, inputs, training=None):
        # Channel attention mechanism (much simpler and faster)
        avg_pool = self.global_avg_pool(inputs)
        fc1 = self.fc1(avg_pool)
        fc2 = self.fc2(fc1)
        attention_weights = self.reshape(fc2)
        
        # Apply attention weights
        output = self.multiply([inputs, attention_weights])
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config


class SEBlock(layers.Layer):
    """Squeeze-and-Excitation block for channel attention"""
    
    def __init__(self, filters: int, reduction: int = 16, name: str = 'se_block', **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.reduction = reduction
        
        self.global_avg_pool = layers.GlobalAveragePooling2D(name=f'{name}_gap')
        self.dense1 = layers.Dense(filters // reduction, activation='relu', name=f'{name}_dense1')
        self.dense2 = layers.Dense(filters, activation='sigmoid', name=f'{name}_dense2')
        self.reshape = layers.Reshape((1, 1, filters), name=f'{name}_reshape')
        self.multiply = layers.Multiply(name=f'{name}_multiply')
    
    def call(self, inputs, training=None):
        # Squeeze: Global average pooling
        squeezed = self.global_avg_pool(inputs)
        
        # Excitation: FC layers
        excited = self.dense1(squeezed)
        excited = self.dense2(excited)
        excited = self.reshape(excited)
        
        # Scale: Element-wise multiplication
        output = self.multiply([inputs, excited])
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'reduction': self.reduction
        })
        return config


class EnhancedBiteMarkCNN:
    """Enhanced CNN architecture for bite mark classification"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize enhanced CNN"""
        self.config = config or ModelConfig()
        self.model = None
        self._setup_mixed_precision()
        
        logger.info(f"EnhancedBiteMarkCNN initialized: {self.config.to_dict()}")
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training for memory efficiency"""
        if self.config.use_mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision enabled (float16)")
            except Exception as e:
                logger.warning(f"Mixed precision setup failed: {e}")
                self.config.use_mixed_precision = False
    
    def _create_conv_block(self, filters: int, kernel_size: int = 3, 
                          strides: int = 1, use_attention: bool = False,
                          use_se: bool = False, name: str = 'conv_block') -> layers.Layer:
        """Create optimized convolutional block for faster CPU training"""
        def conv_block(x):
            # Regular convolution for better CPU performance
            x = layers.Conv2D(
                filters, kernel_size, strides=strides, padding='same',
                use_bias=False, name=f'{name}_conv',
                kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization)
            )(x)
            
            # Batch normalization
            x = layers.BatchNormalization(name=f'{name}_bn')(x)
            
            # Faster activation
            x = layers.Activation('relu', name=f'{name}_activation')(x)
            
            # Squeeze-and-Excitation block (optional)
            if use_se:
                x = SEBlock(filters, name=f'{name}_se')(x)
            
            # Self-attention module (optional)
            if use_attention:
                x = AttentionModule(filters, name=f'{name}_attention')(x)
            
            return x
        
        return conv_block
    
    def _create_enhanced_cnn(self) -> tf.keras.Model:
        """Create highly optimized enhanced CNN for fastest training"""
        inputs = layers.Input(shape=self.config.input_shape, name='input')
        
        # Very lightweight initial feature extraction
        x = layers.Conv2D(
            self.config.base_filters, 3, strides=1, padding='same', 
            use_bias=False, name='initial_conv',
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization)
        )(inputs)
        x = layers.BatchNormalization(name='initial_bn')(x)
        x = layers.Activation('relu', name='initial_activation')(x)
        
        # Stage 1: Minimal feature extraction
        current_filters = self.config.base_filters
        x = self._create_conv_block(
            current_filters, kernel_size=3, use_se=False, name='stage1_block1'
        )(x)
        x = layers.MaxPooling2D(2, strides=2, name='pool1')(x)
        
        # Stage 2: Single attention block for efficiency
        current_filters = int(current_filters * 1.5)
        x = self._create_conv_block(
            current_filters, kernel_size=3, 
            use_attention=self.config.use_attention, use_se=False, name='stage2_block1'
        )(x)
        x = layers.MaxPooling2D(2, strides=2, name='pool2')(x)
        
        # Stage 3: Final feature extraction (minimal)
        current_filters = int(current_filters * 1.2)  # Reduced growth
        x = self._create_conv_block(
            current_filters, kernel_size=3, use_se=False, name='stage3_block1'
        )(x)
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Ultra-simple classification head
        x = layers.Dropout(self.config.dropout_rate, name='dropout1')(x)
        x = layers.Dense(
            64, activation='relu', name='dense1',  # Reduced from 128
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization)
        )(x)
        
        # Output layer
        if self.config.use_mixed_precision:
            x = layers.Dense(
                self.config.num_classes, activation='softmax', 
                name='predictions', dtype='float32',
                kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization)
            )(x)
        else:
            x = layers.Dense(
                self.config.num_classes, activation='softmax', 
                name='predictions',
                kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization)
            )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=x, name='enhanced_bitemark_cnn')
        return model
    
    def _create_mobilenet_model(self) -> tf.keras.Model:
        """Create MobileNetV3 based model"""
        # Load pre-trained MobileNetV3
        base_model = MobileNetV3Small(
            weights='imagenet',
            include_top=False,
            input_shape=self.config.input_shape
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        inputs = layers.Input(shape=self.config.input_shape, name='input')
        
        # Base feature extraction
        x = base_model(inputs, training=False)
        
        # Add attention if requested
        if self.config.use_attention:
            x = AttentionModule(x.shape[-1], name='attention')(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Classification head
        x = layers.Dropout(self.config.dropout_rate, name='dropout1')(x)
        x = layers.Dense(
            128, activation='swish', name='dense1',
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization)
        )(x)
        x = layers.Dropout(self.config.dropout_rate / 2, name='dropout2')(x)
        
        # Output layer
        if self.config.use_mixed_precision:
            x = layers.Dense(
                self.config.num_classes, activation='softmax', 
                name='predictions', dtype='float32'
            )(x)
        else:
            x = layers.Dense(
                self.config.num_classes, activation='softmax', name='predictions'
            )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=x, name='mobilenet_bitemark')
        return model
    
    def _create_efficientnet_model(self) -> tf.keras.Model:
        """Create EfficientNet based model"""
        # Load pre-trained EfficientNet
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.config.input_shape
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        inputs = layers.Input(shape=self.config.input_shape, name='input')
        
        # Base feature extraction
        x = base_model(inputs, training=False)
        
        # Add attention if requested
        if self.config.use_attention:
            x = AttentionModule(x.shape[-1], name='attention')(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Classification head
        x = layers.Dropout(self.config.dropout_rate, name='dropout1')(x)
        x = layers.Dense(
            256, activation='swish', name='dense1',
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization)
        )(x)
        x = layers.Dropout(self.config.dropout_rate / 2, name='dropout2')(x)
        
        # Output layer
        if self.config.use_mixed_precision:
            x = layers.Dense(
                self.config.num_classes, activation='softmax', 
                name='predictions', dtype='float32'
            )(x)
        else:
            x = layers.Dense(
                self.config.num_classes, activation='softmax', name='predictions'
            )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=x, name='efficientnet_bitemark')
        return model
    
    def build_model(self, num_classes: Optional[int] = None) -> tf.keras.Model:
        """Build the complete model"""
        if num_classes:
            self.config.num_classes = num_classes
        
        # Create model based on type
        if self.config.model_type == 'enhanced_cnn':
            self.model = self._create_enhanced_cnn()
        elif self.config.model_type == 'mobilenet':
            self.model = self._create_mobilenet_model()
        elif self.config.model_type == 'efficientnet':
            self.model = self._create_efficientnet_model()
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        logger.info(f"Model built: {self.config.model_type}")
        logger.info(f"Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def compile_model(self):
        """Compile model with optimizer and loss"""
        if self.model is None:
            raise ValueError("Model must be built before compiling")
        
        # Setup optimizer
        if self.config.optimizer == 'adam':
            optimizer = optimizers.Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'adamw':
            optimizer = optimizers.AdamW(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.l2_regularization
            )
        elif self.config.optimizer == 'sgd':
            optimizer = optimizers.SGD(
                learning_rate=self.config.learning_rate,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        logger.info(f"Model compiled with {self.config.optimizer} optimizer")
    
    def freeze_backbone(self):
        """Freeze backbone layers for transfer learning"""
        if self.config.model_type in ['mobilenet', 'efficientnet']:
            # Find the base model layer
            for layer in self.model.layers:
                if layer.name in ['mobilenetv3small', 'efficientnetb0']:
                    layer.trainable = False
                    logger.info(f"Frozen backbone: {layer.name}")
                    break
    
    def unfreeze_backbone(self):
        """Unfreeze backbone layers for fine-tuning"""
        if self.config.model_type in ['mobilenet', 'efficientnet']:
            # Find the base model layer
            for layer in self.model.layers:
                if layer.name in ['mobilenetv3small', 'efficientnetb0']:
                    layer.trainable = True
                    logger.info(f"Unfrozen backbone: {layer.name}")
                    break
    
    def set_learning_rate(self, learning_rate: float):
        """Set learning rate for fine-tuning"""
        if self.model is None:
            raise ValueError("Model must be built and compiled")
        
        self.model.optimizer.learning_rate = learning_rate
        self.config.learning_rate = learning_rate
        logger.info(f"Learning rate set to: {learning_rate}")
    
    def calculate_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for imbalanced datasets"""
        from sklearn.utils.class_weight import compute_class_weight
        
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_train
        )
        
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        logger.info("Class weights calculated:")
        for class_id, weight in class_weight_dict.items():
            count = np.sum(y_train == class_id)
            logger.info(f"  Class {class_id}: {weight:.3f} (count: {count})")
        
        return class_weight_dict
    
    def create_advanced_callbacks(self, model_dir: str, output_dir: str,
                                monitor_metric: str = 'val_accuracy') -> List[tf.keras.callbacks.Callback]:
        """Create advanced callbacks for training"""
        callbacks_list = []
        
        # Model checkpoint
        checkpoint_path = os.path.join(model_dir, 'best_model.h5')
        checkpoint = callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor=monitor_metric,
            save_best_only=True,
            save_weights_only=False,
            mode='max' if 'acc' in monitor_metric else 'min',
            verbose=1
        )
        callbacks_list.append(checkpoint)
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=15,
            restore_best_weights=True,
            mode='max' if 'acc' in monitor_metric else 'min',
            verbose=1
        )
        callbacks_list.append(early_stopping)
        
        # Reduce learning rate on plateau
        lr_reducer = callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            mode='max' if 'acc' in monitor_metric else 'min',
            verbose=1
        )
        callbacks_list.append(lr_reducer)
        
        # TensorBoard logging
        log_dir = os.path.join(output_dir, 'logs')
        tensorboard = callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch='100, 110'
        )
        callbacks_list.append(tensorboard)
        
        # Custom learning rate scheduler
        def schedule(epoch, lr):
            if epoch < 10:
                return lr
            elif epoch < 20:
                return lr * 0.9
            else:
                return lr * 0.95
        
        lr_scheduler = callbacks.LearningRateScheduler(schedule, verbose=1)
        callbacks_list.append(lr_scheduler)
        
        logger.info(f"Created {len(callbacks_list)} advanced callbacks")
        return callbacks_list
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        if self.model is None:
            raise ValueError("Model must be built first")
        
        return {
            'model_type': self.config.model_type,
            'total_parameters': self.model.count_params(),
            'trainable_parameters': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
            'input_shape': self.config.input_shape,
            'num_classes': self.config.num_classes,
            'use_attention': self.config.use_attention,
            'use_mixed_precision': self.config.use_mixed_precision,
            'config': self.config.to_dict()
        }