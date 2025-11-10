"""
Data Preprocessing Module for Bite Mark Classification
Handles image loading, resizing, normalization, and dataset splitting
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path


class BiteMarkPreprocessor:
    """Preprocessor for bite mark images"""
    
    def __init__(self, img_size=(224, 224), grayscale=False, normalize=True, 
                 adaptive_histogram=True, denoise=True):
        """
        Initialize preprocessor
        
        Args:
            img_size: Target image size (height, width)
            grayscale: Convert to grayscale if True (False recommended for real data)
            normalize: Normalize pixel values to [0, 1] if True
            adaptive_histogram: Apply CLAHE for better contrast
            denoise: Apply denoising filter
        """
        self.img_size = img_size
        self.grayscale = grayscale
        self.normalize = normalize
        self.adaptive_histogram = adaptive_histogram
        self.denoise = denoise
        # Only include classes that have data
        self.class_names = ['human', 'dog', 'snake']  # Removed 'cat' (no data)
        
    def load_sample_data(self, data_dir='data/raw', save_processed=False, processed_dir='data/processed'):
        """
        Load sample data or create synthetic data if no real data exists
        
        Args:
            data_dir: Directory containing class folders
            save_processed: If True, save preprocessed images to disk
            processed_dir: Directory to save processed images
            
        Returns:
            images, labels, class_names
        """
        images = []
        labels = []
        
        # Check if data exists
        total_files = 0
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(data_dir, class_name)
            if os.path.exists(class_path):
                files = [f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                total_files += len(files)
        
        if total_files == 0:
            print("⚠ No real data found. Generating synthetic dataset for demonstration...")
            return self._generate_synthetic_data()
        
        # Create processed directory if saving
        if save_processed:
            os.makedirs(processed_dir, exist_ok=True)
            for class_name in self.class_names:
                os.makedirs(os.path.join(processed_dir, class_name), exist_ok=True)
        
        # Load real data
        print(f"📂 Loading data from {data_dir}...")
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                continue
                
            files = [f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            for file in files:
                img_path = os.path.join(class_path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = self._preprocess_image(img)
                    images.append(img)
                    labels.append(class_idx)
                    
                    # Save processed image if requested
                    if save_processed:
                        output_path = os.path.join(processed_dir, class_name, file)
                        # Convert back to uint8 for saving
                        if self.normalize:
                            save_img = (img * 255).astype(np.uint8)
                        else:
                            save_img = img.astype(np.uint8)
                        cv2.imwrite(output_path, save_img)
            
            print(f"  ✓ Loaded {len(files)} images from class '{class_name}'")
        
        if save_processed:
            print(f"\n💾 Saved {len(images)} processed images to {processed_dir}")
        
        return np.array(images), np.array(labels), self.class_names
    
    def _preprocess_image(self, img):
        """Apply preprocessing to a single image with real data optimizations"""
        
        # Denoise if enabled (reduce noise from real photos)
        if self.denoise and len(img.shape) == 3:
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        elif self.denoise and len(img.shape) == 2:
            img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        
        # Convert color space if needed
        if self.grayscale and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif not self.grayscale and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Apply CLAHE for adaptive histogram equalization (better contrast)
        if self.adaptive_histogram:
            if len(img.shape) == 3 and not self.grayscale:
                # Convert to LAB color space for better contrast enhancement
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                img = cv2.merge([l, a, b])
                img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
            elif len(img.shape) == 2 or self.grayscale:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img = clahe.apply(img)
        
        # Resize with high-quality interpolation
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize to [0, 1]
        if self.normalize:
            img = img.astype(np.float32) / 255.0
        
        # Add channel dimension if grayscale
        if self.grayscale and len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        
        return img
    
    def _generate_synthetic_data(self, samples_per_class=200):
        """
        Generate synthetic bite mark-like data for demonstration
        Creates patterns that simulate different bite mark characteristics
        """
        print(f"  Generating {samples_per_class} samples per class...")
        
        images = []
        labels = []
        
        np.random.seed(42)
        
        for class_idx, class_name in enumerate(self.class_names):
            for i in range(samples_per_class):
                img = self._create_synthetic_bite_mark(class_idx)
                images.append(img)
                labels.append(class_idx)
            
            print(f"  ✓ Generated {samples_per_class} synthetic '{class_name}' samples")
        
        return np.array(images), np.array(labels), self.class_names
    
    def _create_synthetic_bite_mark(self, class_idx):
        """Create a synthetic bite mark pattern based on class"""
        img = np.ones((*self.img_size, 1 if self.grayscale else 3), dtype=np.float32) * 0.9
        
        # Different patterns for different classes
        if class_idx == 0:  # Human
            # Rectangular bite pattern with individual teeth marks
            self._add_rectangular_bite(img, num_marks=12, spacing=15)
        elif class_idx == 1:  # Cat
            # Small, sharp puncture marks
            self._add_puncture_marks(img, num_marks=4, size=3)
        elif class_idx == 2:  # Dog
            # Larger bite with curved pattern
            self._add_curved_bite(img, num_marks=10, spacing=20)
        else:  # Snake
            # Two fang marks
            self._add_fang_marks(img)
        
        # Add noise and variations
        noise = np.random.normal(0, 0.05, img.shape)
        img = np.clip(img + noise, 0, 1)
        
        return img.astype(np.float32)
    
    def _add_rectangular_bite(self, img, num_marks=12, spacing=15):
        """Add rectangular bite pattern (human-like)"""
        h, w = self.img_size
        center_y, center_x = h // 2, w // 2
        
        # Upper and lower teeth
        for i in range(num_marks // 2):
            x = center_x - (num_marks // 4) * spacing + i * spacing
            # Upper teeth
            cv2.circle(img, (x, center_y - 20), 
                      radius=np.random.randint(3, 6), 
                      color=0.2, thickness=-1)
            # Lower teeth
            cv2.circle(img, (x, center_y + 20), 
                      radius=np.random.randint(3, 6), 
                      color=0.2, thickness=-1)
    
    def _add_puncture_marks(self, img, num_marks=4, size=3):
        """Add small puncture marks (cat-like)"""
        h, w = self.img_size
        center_y, center_x = h // 2, w // 2
        
        # Random positions around center
        for _ in range(num_marks):
            offset_x = np.random.randint(-30, 30)
            offset_y = np.random.randint(-30, 30)
            cv2.circle(img, (center_x + offset_x, center_y + offset_y),
                      radius=size, color=0.1, thickness=-1)
    
    def _add_curved_bite(self, img, num_marks=10, spacing=20):
        """Add curved bite pattern (dog-like)"""
        h, w = self.img_size
        center_y, center_x = h // 2, w // 2
        
        # Create curved pattern
        for i in range(num_marks):
            angle = (i / num_marks) * np.pi
            x = int(center_x + np.cos(angle) * spacing * 2)
            y = int(center_y + np.sin(angle) * spacing)
            cv2.circle(img, (x, y), 
                      radius=np.random.randint(4, 7), 
                      color=0.15, thickness=-1)
    
    def _add_fang_marks(self, img):
        """Add two fang marks (snake-like)"""
        h, w = self.img_size
        center_y, center_x = h // 2, w // 2
        
        # Two prominent fangs
        cv2.circle(img, (center_x - 15, center_y), 
                  radius=5, color=0.0, thickness=-1)
        cv2.circle(img, (center_x + 15, center_y), 
                  radius=5, color=0.0, thickness=-1)
        
        # Add venom trail effect
        cv2.line(img, (center_x - 15, center_y), 
                (center_x - 15, center_y + 30), 
                color=0.3, thickness=2)
        cv2.line(img, (center_x + 15, center_y), 
                (center_x + 15, center_y + 30), 
                color=0.3, thickness=2)
    
    def create_tf_dataset(self, images, labels, batch_size=32, 
                         shuffle=True, augment=False):
        """
        Create TensorFlow dataset with optional augmentation
        
        Args:
            images: NumPy array of images
            labels: NumPy array of labels
            batch_size: Batch size for training
            shuffle: Whether to shuffle the dataset
            augment: Whether to apply data augmentation
            
        Returns:
            tf.data.Dataset
        """
        # Convert labels to one-hot encoding
        labels_onehot = keras.utils.to_categorical(labels, num_classes=len(self.class_names))
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, labels_onehot))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(images))
        
        if augment:
            dataset = dataset.map(self._augment_fn, 
                                 num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _augment_fn(self, image, label):
        """Data augmentation function for tf.data pipeline"""
        # Random flip
        image = tf.image.random_flip_left_right(image)
        
        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.1)
        
        # Random contrast
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        return image, label
    
    def split_data(self, images, labels, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, validation, and test sets
        
        Args:
            images: NumPy array of images
            labels: NumPy array of labels
            test_size: Proportion of test set
            val_size: Proportion of validation set (from training data)
            random_state: Random seed for reproducibility
            
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_size, 
            random_state=random_state, stratify=labels
        )
        
        # Second split: separate validation from training
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio,
            random_state=random_state, stratify=y_temp
        )
        
        print(f"\n📊 Dataset Split:")
        print(f"  Training:   {len(X_train)} samples ({len(X_train)/len(images)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(images)*100:.1f}%)")
        print(f"  Test:       {len(X_test)} samples ({len(X_test)/len(images)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """Test preprocessing module"""
    print("🔧 Testing Data Preprocessing Module...")
    
    preprocessor = BiteMarkPreprocessor(img_size=(224, 224), grayscale=True)
    images, labels, class_names = preprocessor.load_sample_data()
    
    print(f"\n✓ Loaded {len(images)} images")
    print(f"  Image shape: {images[0].shape}")
    print(f"  Label distribution: {np.bincount(labels)}")
    print(f"  Classes: {class_names}")
    
    # Test data splitting
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(images, labels)
    
    # Test TF dataset creation
    train_dataset = preprocessor.create_tf_dataset(X_train, y_train, batch_size=16, augment=True)
    print(f"\n✓ TensorFlow dataset created successfully")


if __name__ == "__main__":
    main()
