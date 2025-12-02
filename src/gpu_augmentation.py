"""
⚡ GPU-ACCELERATED DATA AUGMENTATION
Advanced augmentation pipeline optimized for GPU processing

Features:
- GPU-based augmentation operations for maximum speed
- Forensic-specific augmentation preserving bite mark features
- Intelligent class balancing to handle imbalanced datasets
- Deterministic augmentation with reproducible seeds
- Real-time augmentation during training
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for GPU-based augmentation"""
    # Core settings
    use_gpu: bool = True
    preserve_forensic_features: bool = True
    balance_classes: bool = True
    seed: int = 42
    
    # Augmentation intensities
    rotation_range: float = 15.0
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    shear_range: float = 0.1
    zoom_range: Tuple[float, float] = (0.9, 1.1)
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    
    # Advanced augmentations
    use_elastic_transform: bool = True
    use_perspective_transform: bool = True
    use_gaussian_noise: bool = True
    use_motion_blur: bool = True
    
    # GPU optimization
    batch_size: int = 32
    num_parallel_calls: int = tf.data.AUTOTUNE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'use_gpu': self.use_gpu,
            'preserve_forensic_features': self.preserve_forensic_features,
            'balance_classes': self.balance_classes,
            'rotation_range': self.rotation_range,
            'width_shift_range': self.width_shift_range,
            'height_shift_range': self.height_shift_range,
            'shear_range': self.shear_range,
            'zoom_range': self.zoom_range,
            'brightness_range': self.brightness_range,
            'contrast_range': self.contrast_range
        }


class GPUAugmentor:
    """GPU-accelerated data augmentation with forensic preservation"""
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """Initialize GPU augmentor"""
        self.config = config or AugmentationConfig()
        self._setup_gpu_augmentation()
        self._setup_augmentation_layers()
        
        logger.info(f"GPUAugmentor initialized: GPU={self.config.use_gpu}")
    
    def _setup_gpu_augmentation(self):
        """Setup GPU-based augmentation pipeline"""
        if self.config.use_gpu:
            # Check GPU availability
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU augmentation enabled: {len(gpus)} GPUs available")
                except RuntimeError as e:
                    logger.warning(f"GPU setup failed: {e}")
                    self.config.use_gpu = False
            else:
                logger.warning("No GPU found, falling back to CPU augmentation")
                self.config.use_gpu = False
    
    def _setup_augmentation_layers(self):
        """Setup TensorFlow augmentation layers for GPU processing"""
        self.augmentation_layers = []
        
        # Core geometric transformations (preserve bite mark structure)
        if self.config.preserve_forensic_features:
            # Mild transformations that preserve forensic characteristics
            self.augmentation_layers.extend([
                tf.keras.layers.RandomRotation(
                    factor=self.config.rotation_range / 360.0,
                    fill_mode='reflect',
                    interpolation='bilinear',
                    seed=self.config.seed
                ),
                tf.keras.layers.RandomTranslation(
                    height_factor=self.config.height_shift_range,
                    width_factor=self.config.width_shift_range,
                    fill_mode='reflect',
                    interpolation='bilinear',
                    seed=self.config.seed + 1
                ),
                tf.keras.layers.RandomZoom(
                    height_factor=(-0.1, 0.1),
                    width_factor=(-0.1, 0.1),
                    fill_mode='reflect',
                    interpolation='bilinear',
                    seed=self.config.seed + 2
                )
            ])
        else:
            # More aggressive transformations for general datasets
            self.augmentation_layers.extend([
                tf.keras.layers.RandomRotation(
                    factor=self.config.rotation_range / 360.0,
                    fill_mode='reflect',
                    seed=self.config.seed
                ),
                tf.keras.layers.RandomTranslation(
                    height_factor=self.config.height_shift_range,
                    width_factor=self.config.width_shift_range,
                    fill_mode='reflect',
                    seed=self.config.seed + 1
                ),
                tf.keras.layers.RandomZoom(
                    height_factor=(1.0 - self.config.zoom_range[1], 
                                 self.config.zoom_range[1] - 1.0),
                    width_factor=(1.0 - self.config.zoom_range[1], 
                                self.config.zoom_range[1] - 1.0),
                    fill_mode='reflect',
                    seed=self.config.seed + 2
                )
            ])
        
        # Photometric transformations
        self.augmentation_layers.extend([
            tf.keras.layers.RandomBrightness(
                factor=0.2,
                value_range=(0.0, 1.0),
                seed=self.config.seed + 3
            ),
            tf.keras.layers.RandomContrast(
                factor=0.2,
                seed=self.config.seed + 4
            )
        ])
        
        # Additional augmentations for robustness
        if not self.config.preserve_forensic_features:
            self.augmentation_layers.append(
                tf.keras.layers.RandomFlip(
                    mode="horizontal",
                    seed=self.config.seed + 5
                )
            )
        
        # Combine all layers into a sequential model
        self.augmentation_model = tf.keras.Sequential(
            self.augmentation_layers,
            name="gpu_augmentation_pipeline"
        )
    
    @tf.function
    def _apply_gpu_augmentation(self, images: tf.Tensor) -> tf.Tensor:
        """Apply GPU-based augmentation to batch of images"""
        return self.augmentation_model(images, training=True)
    
    def _apply_advanced_augmentations(self, images: np.ndarray) -> np.ndarray:
        """Apply advanced CPU-based augmentations"""
        augmented_images = []
        
        for img in images:
            try:
                augmented = img.copy()
                
                # Elastic transformation (mild for forensics)
                if (self.config.use_elastic_transform and 
                    np.random.random() < 0.3):
                    augmented = self._elastic_transform(augmented, alpha=0.5, sigma=0.1)
                
                # Perspective transformation (very mild)
                if (self.config.use_perspective_transform and 
                    self.config.preserve_forensic_features and 
                    np.random.random() < 0.2):
                    augmented = self._perspective_transform(augmented, strength=0.05)
                
                # Gaussian noise (minimal)
                if (self.config.use_gaussian_noise and 
                    np.random.random() < 0.3):
                    noise_std = 0.01 if self.config.preserve_forensic_features else 0.02
                    noise = np.random.normal(0, noise_std, augmented.shape)
                    augmented = np.clip(augmented + noise, 0.0, 1.0)
                
                # Motion blur (very mild)
                if (self.config.use_motion_blur and 
                    np.random.random() < 0.2):
                    kernel_size = 3 if self.config.preserve_forensic_features else 5
                    augmented = self._motion_blur(augmented, kernel_size)
                
                augmented_images.append(augmented)
                
            except Exception as e:
                logger.warning(f"Advanced augmentation failed: {e}, using original")
                augmented_images.append(img)
        
        return np.array(augmented_images)
    
    def _elastic_transform(self, image: np.ndarray, alpha: float = 1.0, 
                          sigma: float = 0.1) -> np.ndarray:
        """Apply elastic transformation"""
        try:
            h, w = image.shape[:2]
            
            # Generate displacement fields
            dx = cv2.GaussianBlur(
                np.random.rand(h, w) * 2 - 1, 
                ksize=(0, 0), 
                sigmaX=sigma * h, 
                sigmaY=sigma * w
            ) * alpha * h
            
            dy = cv2.GaussianBlur(
                np.random.rand(h, w) * 2 - 1, 
                ksize=(0, 0), 
                sigmaX=sigma * h, 
                sigmaY=sigma * w
            ) * alpha * w
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)
            
            # Apply transformation
            transformed = cv2.remap(
                image, map_x, map_y, 
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            
            return transformed
            
        except Exception as e:
            logger.warning(f"Elastic transform failed: {e}")
            return image
    
    def _perspective_transform(self, image: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Apply mild perspective transformation"""
        try:
            h, w = image.shape[:2]
            
            # Define source points (corners)
            src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            
            # Add random displacement to corners
            displacement = strength * min(h, w)
            dst_points = src_points + np.random.uniform(
                -displacement, displacement, src_points.shape
            ).astype(np.float32)
            
            # Ensure points are within bounds
            dst_points[:, 0] = np.clip(dst_points[:, 0], 0, w)
            dst_points[:, 1] = np.clip(dst_points[:, 1], 0, h)
            
            # Calculate transformation matrix
            transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply transformation
            transformed = cv2.warpPerspective(
                image, transform_matrix, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            
            return transformed
            
        except Exception as e:
            logger.warning(f"Perspective transform failed: {e}")
            return image
    
    def _motion_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply motion blur"""
        try:
            # Random direction for motion blur
            angle = np.random.uniform(0, 180)
            
            # Create motion blur kernel
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            
            # Rotate kernel
            center = ((kernel_size - 1) / 2, (kernel_size - 1) / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
            
            # Apply blur
            blurred = cv2.filter2D(image, -1, kernel)
            
            return blurred
            
        except Exception as e:
            logger.warning(f"Motion blur failed: {e}")
            return image
    
    def calculate_class_balance(self, labels: np.ndarray, class_names: List[str]) -> Dict[int, float]:
        """Calculate class balancing factors"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        
        # Calculate inverse frequency weights
        class_weights = {}
        max_count = max(counts)
        
        for label, count in zip(unique_labels, counts):
            if self.config.balance_classes:
                # Calculate how much to augment this class
                target_count = max_count
                augmentation_factor = target_count / count
                class_weights[label] = augmentation_factor
            else:
                class_weights[label] = 1.0
        
        logger.info("Class balancing factors:")
        for label, factor in class_weights.items():
            class_name = class_names[label] if label < len(class_names) else f"Class_{label}"
            original_count = counts[label]
            new_count = int(original_count * factor)
            logger.info(f"  {class_name}: {original_count} → {new_count} samples (×{factor:.1f})")
        
        return class_weights
    
    def augment_dataset(self, X_train: np.ndarray, y_train: np.ndarray,
                       augmentation_factor: int = 2, class_names: List[str] = None,
                       save_augmented: bool = False, augmented_dir: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation to training dataset with class balancing"""
        logger.info(f"Starting GPU-accelerated augmentation (factor: {augmentation_factor})")
        
        class_names = class_names or [f"class_{i}" for i in range(len(np.unique(y_train)))]
        
        # Calculate class balancing
        class_weights = self.calculate_class_balance(y_train, class_names)
        
        augmented_images = []
        augmented_labels = []
        
        # Process each class separately for balanced augmentation
        unique_classes = np.unique(y_train)
        
        for class_idx in unique_classes:
            # Get samples for this class
            class_mask = y_train == class_idx
            class_images = X_train[class_mask]
            class_labels = y_train[class_mask]
            
            # Calculate augmentation amount for this class
            class_factor = int(class_weights[class_idx] * augmentation_factor)
            total_augmentations = class_factor - 1  # Subtract 1 for original samples
            
            logger.info(f"Augmenting class {class_names[class_idx]}: "
                       f"{len(class_images)} → {len(class_images) * class_factor} samples")
            
            # Add original samples
            augmented_images.extend(class_images)
            augmented_labels.extend(class_labels)
            
            # Apply augmentations in batches
            if total_augmentations > 0:
                for aug_round in range(total_augmentations):
                    if self.config.use_gpu:
                        # GPU-based augmentation
                        class_images_tf = tf.convert_to_tensor(class_images, dtype=tf.float32)
                        augmented_batch = self._apply_gpu_augmentation(class_images_tf)
                        augmented_batch = augmented_batch.numpy()
                    else:
                        # CPU-based augmentation
                        augmented_batch = class_images.copy()
                    
                    # Apply advanced CPU augmentations
                    if (self.config.use_elastic_transform or 
                        self.config.use_perspective_transform or
                        self.config.use_gaussian_noise or 
                        self.config.use_motion_blur):
                        augmented_batch = self._apply_advanced_augmentations(augmented_batch)
                    
                    augmented_images.extend(augmented_batch)
                    augmented_labels.extend(class_labels)
            
            # Save augmented samples if requested
            if save_augmented and augmented_dir:
                self._save_augmented_class(
                    augmented_batch if total_augmentations > 0 else class_images,
                    class_names[class_idx],
                    augmented_dir,
                    aug_round=aug_round if total_augmentations > 0 else 0
                )
        
        # Convert to numpy arrays
        X_augmented = np.array(augmented_images)
        y_augmented = np.array(augmented_labels)
        
        # Shuffle the augmented dataset
        shuffle_indices = np.random.permutation(len(X_augmented))
        X_augmented = X_augmented[shuffle_indices]
        y_augmented = y_augmented[shuffle_indices]
        
        logger.info(f"Augmentation completed: {len(X_train)} → {len(X_augmented)} samples")
        
        # Print final class distribution
        unique_labels, counts = np.unique(y_augmented, return_counts=True)
        logger.info("Final class distribution:")
        for label, count in zip(unique_labels, counts):
            class_name = class_names[label]
            percentage = (count / len(y_augmented)) * 100
            logger.info(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        return X_augmented, y_augmented
    
    def _save_augmented_class(self, images: np.ndarray, class_name: str, 
                             augmented_dir: str, aug_round: int = 0):
        """Save augmented images for a class"""
        try:
            class_aug_dir = Path(augmented_dir) / class_name
            class_aug_dir.mkdir(parents=True, exist_ok=True)
            
            for i, img in enumerate(images):
                # Convert back to uint8 for saving
                img_uint8 = (img * 255).astype(np.uint8)
                save_path = class_aug_dir / f"aug_{aug_round:02d}_{i:04d}.png"
                
                if len(img.shape) == 3 and img.shape[2] == 3:
                    # RGB image - convert to BGR for OpenCV
                    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(save_path), img_bgr)
                else:
                    # Grayscale image
                    cv2.imwrite(str(save_path), img_uint8)
                    
        except Exception as e:
            logger.warning(f"Failed to save augmented images for {class_name}: {e}")
    
    def create_augmentation_pipeline(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Create real-time augmentation pipeline for training"""
        def augment_batch(images, labels):
            """Augment batch of images"""
            augmented_images = self._apply_gpu_augmentation(images)
            return augmented_images, labels
        
        # Apply augmentation with parallel processing
        augmented_dataset = dataset.map(
            augment_batch,
            num_parallel_calls=self.config.num_parallel_calls
        )
        
        return augmented_dataset
    
    def get_augmentation_summary(self) -> Dict[str, Any]:
        """Get summary of augmentation configuration"""
        return {
            'gpu_acceleration': self.config.use_gpu,
            'preserve_forensic_features': self.config.preserve_forensic_features,
            'balance_classes': self.config.balance_classes,
            'augmentation_layers': len(self.augmentation_layers),
            'config': self.config.to_dict()
        }