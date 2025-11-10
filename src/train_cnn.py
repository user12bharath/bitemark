"""
CNN Training Module for Bite Mark Classification
Optimized for 4GB RTX GPU with advanced training strategies
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight

from data_preprocessing import BiteMarkPreprocessor
from augmentation import BiteMarkAugmentor
from utils import setup_gpu, print_section_header, get_class_weights


class BiteMarkCNN:
    """
    Optimized CNN model for bite mark classification
    Designed to run efficiently on 4GB RTX GPU
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=3, model_type='efficient'):
        """
        Initialize CNN model
        
        Args:
            input_shape: Input image shape (height, width, channels) - default RGB
            num_classes: Number of classification classes (3 for human/dog/snake)
            model_type: 'efficient' (custom lightweight) or 'mobilenet' (transfer learning)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = None
        self.history = None
        
    def build_efficient_model(self):
        """
        Build custom efficient CNN optimized for 4GB GPU
        Uses depthwise separable convolutions and efficient architecture
        """
        print("🏗️  Building Efficient Custom CNN...")
        
        model = models.Sequential(name="BiteMarkCNN_Efficient")
        
        # Input layer
        model.add(layers.Input(shape=self.input_shape))
        
        # Block 1: Initial feature extraction
        model.add(layers.Conv2D(32, (3, 3), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
        
        # Block 2: Depthwise separable convolution (memory efficient)
        model.add(layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.3))
        
        # Block 3: Deeper feature extraction
        model.add(layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.3))
        
        # Block 4: High-level features
        model.add(layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dropout(0.4))
        
        # Classification head
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax', dtype='float32'))
        
        self.model = model
        print(f"✓ Model built successfully")
        return model
    
    def build_mobilenet_model(self):
        """
        Build MobileNetV2-based model for transfer learning
        Optimized for RGB real-world images
        """
        print("🏗️  Building MobileNetV2 Transfer Learning Model...")
        
        # Load MobileNetV2 with ImageNet weights (works for RGB)
        base_model = keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet' if self.input_shape[2] == 3 else None,
            pooling='avg'
        )
        
        # Fine-tune top layers
        base_model.trainable = True
        for layer in base_model.layers[:-30]:  # Freeze lower layers
            layer.trainable = False
        
        # Build model
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs, name="BiteMarkCNN_MobileNet")
        print(f"✓ MobileNetV2 model built with transfer learning (ImageNet weights)")
        return self.model
    
    def build(self):
        """Build model based on specified type"""
        if self.model_type == 'mobilenet':
            return self.build_mobilenet_model()
        else:
            return self.build_efficient_model()
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile model with optimizer and loss function
        
        Args:
            learning_rate: Initial learning rate
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print(f"✓ Model compiled with Adam optimizer (lr={learning_rate})")
    
    def get_callbacks(self, model_path='models/best_model.h5'):
        """
        Create training callbacks for optimization
        
        Args:
            model_path: Path to save best model
            
        Returns:
            List of Keras callbacks
        """
        callback_list = [
            # Save best model
            callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping to prevent overfitting
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir='outputs/logs',
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        print("✓ Training callbacks configured:")
        print("  - ModelCheckpoint (best model saving)")
        print("  - EarlyStopping (patience=15)")
        print("  - ReduceLROnPlateau (factor=0.5, patience=5)")
        print("  - TensorBoard logging")
        
        return callback_list
    
    def train(self, train_data, val_data, epochs=100, class_weights=None, 
              callbacks_list=None):
        """
        Train the model
        
        Args:
            train_data: Training dataset (tf.data.Dataset)
            val_data: Validation dataset (tf.data.Dataset)
            epochs: Maximum number of epochs
            class_weights: Dictionary of class weights for imbalanced data
            callbacks_list: List of Keras callbacks
            
        Returns:
            Training history
        """
        print_section_header("TRAINING PHASE")
        print(f"Maximum epochs: {epochs}")
        print(f"Class weights: {'Enabled' if class_weights else 'Disabled'}")
        
        start_time = time.time()
        
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks_list,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        return self.history, training_time
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
            
            # Calculate model size
            total_params = self.model.count_params()
            print(f"\n📊 Model Statistics:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Estimated size: ~{total_params * 4 / (1024**2):.2f} MB (FP32)")
            print(f"  With Mixed Precision: ~{total_params * 2 / (1024**2):.2f} MB (FP16)")


def main():
    """Main training pipeline"""
    print_section_header("BITE MARK CLASSIFICATION - TRAINING PIPELINE")
    print("Optimized for 4GB RTX GPU")
    
    # Setup GPU
    gpu_available = setup_gpu()
    
    # Configuration
    IMG_SIZE = (224, 224)
    GRAYSCALE = True
    BATCH_SIZE = 16 if gpu_available else 8  # Adaptive batch size
    EPOCHS = 100
    LEARNING_RATE = 0.001
    AUGMENTATION_FACTOR = 2
    
    print(f"\n⚙️  Configuration:")
    print(f"  Image Size: {IMG_SIZE}")
    print(f"  Grayscale: {GRAYSCALE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Max Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Augmentation Factor: {AUGMENTATION_FACTOR}")
    
    # Step 1: Load and preprocess data
    print_section_header("STEP 1: DATA PREPROCESSING")
    preprocessor = BiteMarkPreprocessor(img_size=IMG_SIZE, grayscale=GRAYSCALE)
    images, labels, class_names = preprocessor.load_sample_data()
    
    # Step 2: Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        images, labels, test_size=0.2, val_size=0.1
    )
    
    # Step 3: Data augmentation
    print_section_header("STEP 2: DATA AUGMENTATION")
    augmentor = BiteMarkAugmentor(preserve_features=True)
    X_train_aug, y_train_aug = augmentor.augment_dataset(
        X_train, y_train, augmentation_factor=AUGMENTATION_FACTOR
    )
    
    # Step 4: Create TensorFlow datasets
    print_section_header("STEP 3: CREATING TF DATASETS")
    train_dataset = preprocessor.create_tf_dataset(
        X_train_aug, y_train_aug, batch_size=BATCH_SIZE, shuffle=True, augment=True
    )
    val_dataset = preprocessor.create_tf_dataset(
        X_val, y_val, batch_size=BATCH_SIZE, shuffle=False, augment=False
    )
    test_dataset = preprocessor.create_tf_dataset(
        X_test, y_test, batch_size=BATCH_SIZE, shuffle=False, augment=False
    )
    
    print(f"✓ Datasets created successfully")
    
    # Step 5: Calculate class weights
    class_weights = get_class_weights(y_train_aug)
    
    # Step 6: Build and compile model
    print_section_header("STEP 4: MODEL ARCHITECTURE")
    input_shape = (*IMG_SIZE, 1 if GRAYSCALE else 3)
    cnn = BiteMarkCNN(input_shape=input_shape, num_classes=len(class_names), 
                      model_type='efficient')
    cnn.build()
    cnn.summary()
    cnn.compile_model(learning_rate=LEARNING_RATE)
    
    # Step 7: Train model
    callbacks_list = cnn.get_callbacks(model_path='models/best_model.h5')
    history, training_time = cnn.train(
        train_dataset, val_dataset, 
        epochs=EPOCHS, 
        class_weights=class_weights,
        callbacks_list=callbacks_list
    )
    
    # Save training info
    training_info = {
        'config': {
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'epochs': len(history.history['loss']),
            'learning_rate': LEARNING_RATE,
            'augmentation_factor': AUGMENTATION_FACTOR
        },
        'samples': {
            'train': len(X_train_aug),
            'val': len(X_val),
            'test': len(X_test)
        },
        'training_time': training_time
    }
    
    import json
    with open('outputs/training_info.json', 'w') as f:
        json.dump(training_info, f, indent=4)
    
    print_section_header("TRAINING COMPLETE")
    print(f"✓ Best model saved to: models/best_model.h5")
    print(f"✓ Training info saved to: outputs/training_info.json")
    print(f"\n🎯 Next Step: Run evaluate_model.py to assess performance")


if __name__ == "__main__":
    main()
