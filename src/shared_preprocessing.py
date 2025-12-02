"""
🔬 SHARED PREPROCESSING MODULE
Unified preprocessing for both training and inference to ensure consistency

Features:
- Shared preprocessing between train/test/inference
- Advanced image enhancement (CLAHE, denoising)
- GPU-optimized operations where possible
- Robust error handling and validation
- Support for both single images and batch processing
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing operations"""
    img_size: Tuple[int, int] = (224, 224)
    channels: int = 3
    normalize: bool = True
    adaptive_histogram: bool = True
    denoise: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    denoise_strength: int = 10
    preserve_aspect_ratio: bool = False
    interpolation: int = cv2.INTER_LANCZOS4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'img_size': self.img_size,
            'channels': self.channels,
            'normalize': self.normalize,
            'adaptive_histogram': self.adaptive_histogram,
            'denoise': self.denoise,
            'clahe_clip_limit': self.clahe_clip_limit,
            'clahe_grid_size': self.clahe_grid_size,
            'denoise_strength': self.denoise_strength,
            'preserve_aspect_ratio': self.preserve_aspect_ratio
        }


class SharedPreprocessor:
    """Unified preprocessing for training and inference consistency"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize preprocessor with configuration"""
        self.config = config or PreprocessingConfig()
        self.setup_clahe()
        self._validate_config()
        
        logger.info(f"SharedPreprocessor initialized with config: {self.config.to_dict()}")
    
    def _validate_config(self):
        """Validate preprocessing configuration"""
        if self.config.img_size[0] <= 0 or self.config.img_size[1] <= 0:
            raise ValueError("Image size must be positive")
        if self.config.channels not in [1, 3]:
            raise ValueError("Channels must be 1 (grayscale) or 3 (RGB)")
        if self.config.clahe_clip_limit <= 0:
            raise ValueError("CLAHE clip limit must be positive")
    
    def setup_clahe(self):
        """Setup CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        try:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_grid_size
            )
        except Exception as e:
            logger.error(f"Failed to setup CLAHE: {e}")
            self.clahe = None
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to enhance contrast"""
        if not self.config.adaptive_histogram or self.clahe is None:
            return image
        
        try:
            if len(image.shape) == 3:  # RGB image
                # Convert to LAB color space for better CLAHE
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:  # Grayscale image
                enhanced = self.clahe.apply(image)
            
            return enhanced
        except Exception as e:
            logger.warning(f"CLAHE failed: {e}, returning original image")
            return image
    
    def apply_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising"""
        if not self.config.denoise:
            return image
        
        try:
            if len(image.shape) == 3:  # RGB image
                denoised = cv2.fastNlMeansDenoisingColored(
                    image, None, 
                    h=self.config.denoise_strength,
                    hColor=self.config.denoise_strength,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
            else:  # Grayscale image
                denoised = cv2.fastNlMeansDenoising(
                    image, None,
                    h=self.config.denoise_strength,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
            
            return denoised
        except Exception as e:
            logger.warning(f"Denoising failed: {e}, returning original image")
            return image
    
    def resize_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Resize image with optional aspect ratio preservation"""
        target_size = target_size or self.config.img_size
        
        try:
            if self.config.preserve_aspect_ratio:
                # Calculate scaling factor to maintain aspect ratio
                h, w = image.shape[:2]
                target_h, target_w = target_size
                
                scale = min(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                # Resize maintaining aspect ratio
                resized = cv2.resize(image, (new_w, new_h), interpolation=self.config.interpolation)
                
                # Pad to exact target size
                pad_h = (target_h - new_h) // 2
                pad_w = (target_w - new_w) // 2
                
                if len(image.shape) == 3:
                    padded = cv2.copyMakeBorder(
                        resized, pad_h, target_h - new_h - pad_h,
                        pad_w, target_w - new_w - pad_w,
                        cv2.BORDER_CONSTANT, value=[0, 0, 0]
                    )
                else:
                    padded = cv2.copyMakeBorder(
                        resized, pad_h, target_h - new_h - pad_h,
                        pad_w, target_w - new_w - pad_w,
                        cv2.BORDER_CONSTANT, value=0
                    )
                
                return padded
            else:
                # Direct resize without aspect ratio preservation
                return cv2.resize(image, target_size, interpolation=self.config.interpolation)
                
        except Exception as e:
            logger.error(f"Resize failed: {e}")
            raise
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        if not self.config.normalize:
            return image
        
        try:
            # Ensure image is in proper range
            if image.dtype == np.uint8:
                normalized = image.astype(np.float32) / 255.0
            elif image.dtype == np.float32 and image.max() > 1.0:
                normalized = image / 255.0
            else:
                normalized = image.astype(np.float32)
            
            # Clip to valid range
            normalized = np.clip(normalized, 0.0, 1.0)
            return normalized
            
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            raise
    
    def ensure_channels(self, image: np.ndarray) -> np.ndarray:
        """Ensure image has correct number of channels"""
        try:
            current_channels = 1 if len(image.shape) == 2 else image.shape[2]
            
            if current_channels == self.config.channels:
                return image
            
            if current_channels == 1 and self.config.channels == 3:
                # Convert grayscale to RGB
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif current_channels == 3 and self.config.channels == 1:
                # Convert RGB to grayscale
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif current_channels == 4 and self.config.channels == 3:
                # Convert RGBA to RGB
                return image[:, :, :3]
            else:
                logger.warning(f"Unsupported channel conversion: {current_channels} -> {self.config.channels}")
                return image
                
        except Exception as e:
            logger.error(f"Channel conversion failed: {e}")
            raise
    
    def preprocess_single_image(self, image: np.ndarray) -> np.ndarray:
        """Complete preprocessing pipeline for a single image"""
        try:
            # Ensure proper data type
            if image.dtype != np.uint8 and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # Apply denoising first
            processed = self.apply_denoising(image)
            
            # Apply CLAHE for contrast enhancement
            processed = self.apply_clahe(processed)
            
            # Ensure correct channels
            processed = self.ensure_channels(processed)
            
            # Resize to target size
            processed = self.resize_image(processed)
            
            # Normalize to [0, 1] range
            processed = self.normalize_image(processed)
            
            # Ensure proper shape
            if len(processed.shape) == 2:
                processed = np.expand_dims(processed, axis=-1)
            
            return processed
            
        except Exception as e:
            logger.error(f"Single image preprocessing failed: {e}")
            raise
    
    def preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """Preprocess a batch of images"""
        try:
            processed_images = []
            
            for i, image in enumerate(images):
                try:
                    processed = self.preprocess_single_image(image)
                    processed_images.append(processed)
                except Exception as e:
                    logger.warning(f"Failed to preprocess image {i}: {e}")
                    # Create a blank image as fallback
                    blank = np.zeros((*self.config.img_size, self.config.channels), dtype=np.float32)
                    processed_images.append(blank)
            
            return np.array(processed_images)
            
        except Exception as e:
            logger.error(f"Batch preprocessing failed: {e}")
            raise
    
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image from file"""
        try:
            # Load image
            if self.config.channels == 3:
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Preprocess
            processed = self.preprocess_single_image(image)
            
            return processed
            
        except Exception as e:
            logger.error(f"Failed to load and preprocess {image_path}: {e}")
            raise
    
    def load_dataset(self, data_dir: str, save_processed: bool = False, 
                    processed_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load complete dataset with preprocessing"""
        logger.info(f"Loading dataset from {data_dir}")
        
        try:
            images = []
            labels = []
            class_names = []
            
            data_path = Path(data_dir)
            if not data_path.exists():
                raise ValueError(f"Data directory not found: {data_dir}")
            
            # Get class directories
            class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
            class_dirs.sort()  # Ensure consistent ordering
            
            for class_idx, class_dir in enumerate(class_dirs):
                class_name = class_dir.name
                class_names.append(class_name)
                
                logger.info(f"Loading class: {class_name}")
                
                # Get image files
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                    image_files.extend(class_dir.glob(ext))
                    image_files.extend(class_dir.glob(ext.upper()))
                
                class_images = []
                class_labels = []
                
                for img_path in image_files:
                    try:
                        processed_img = self.load_and_preprocess_image(str(img_path))
                        class_images.append(processed_img)
                        class_labels.append(class_idx)
                    except Exception as e:
                        logger.warning(f"Failed to load {img_path}: {e}")
                        continue
                
                logger.info(f"  Loaded {len(class_images)} images for class {class_name}")
                images.extend(class_images)
                labels.extend(class_labels)
                
                # Save processed images if requested
                if save_processed and processed_dir:
                    self._save_processed_class(class_images, class_name, processed_dir)
            
            images = np.array(images)
            labels = np.array(labels)
            
            logger.info(f"Dataset loaded: {len(images)} images, {len(class_names)} classes")
            logger.info(f"Image shape: {images[0].shape}")
            logger.info(f"Classes: {class_names}")
            
            return images, labels, class_names
            
        except Exception as e:
            logger.error(f"Dataset loading failed: {e}")
            raise
    
    def _save_processed_class(self, images: List[np.ndarray], class_name: str, processed_dir: str):
        """Save processed images for a class"""
        try:
            class_processed_dir = Path(processed_dir) / class_name
            class_processed_dir.mkdir(parents=True, exist_ok=True)
            
            for i, img in enumerate(images):
                # Convert back to uint8 for saving
                img_uint8 = (img * 255).astype(np.uint8)
                save_path = class_processed_dir / f"processed_{i:04d}.png"
                
                if self.config.channels == 3:
                    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(save_path), img_bgr)
                else:
                    cv2.imwrite(str(save_path), img_uint8)
                    
        except Exception as e:
            logger.warning(f"Failed to save processed images for {class_name}: {e}")
    
    def split_data(self, images: np.ndarray, labels: np.ndarray, 
                  test_size: float = 0.2, val_size: float = 0.15, 
                  random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """Split data into train, validation, and test sets with stratification"""
        from sklearn.model_selection import train_test_split
        
        try:
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                images, labels, 
                test_size=test_size,
                random_state=random_state,
                stratify=labels
            )
            
            # Second split: separate train and validation
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=y_temp
            )
            
            logger.info(f"Data split completed:")
            logger.info(f"  Train: {len(X_train)} ({len(X_train)/len(images)*100:.1f}%)")
            logger.info(f"  Val: {len(X_val)} ({len(X_val)/len(images)*100:.1f}%)")
            logger.info(f"  Test: {len(X_test)} ({len(X_test)/len(images)*100:.1f}%)")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Data split failed: {e}")
            raise
    
    def create_optimized_dataset(self, images: np.ndarray, labels: np.ndarray,
                               batch_size: int = 32, shuffle: bool = True,
                               augment: bool = False, cache: bool = True,
                               prefetch: bool = True) -> tf.data.Dataset:
        """Create optimized TensorFlow dataset"""
        try:
            # Create dataset from arrays
            dataset = tf.data.Dataset.from_tensor_slices((images, labels))
            
            # Shuffle if requested
            if shuffle:
                buffer_size = min(len(images), 1000)
                dataset = dataset.shuffle(buffer_size=buffer_size)
            
            # Batch the data
            dataset = dataset.batch(batch_size)
            
            # Cache for performance
            if cache:
                dataset = dataset.cache()
            
            # Prefetch for performance
            if prefetch:
                dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            logger.info(f"Created optimized dataset with {len(images)} samples")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Dataset creation failed: {e}")
            raise


# For backward compatibility
class BiteMarkPreprocessor(SharedPreprocessor):
    """Backward compatibility alias"""
    pass


def preprocess_for_api(image_array: np.ndarray, config: Optional[PreprocessingConfig] = None) -> np.ndarray:
    """Convenience function for API preprocessing"""
    preprocessor = SharedPreprocessor(config)
    return preprocessor.preprocess_single_image(image_array)


def preprocess_image_file(image_path: str, config: Optional[PreprocessingConfig] = None) -> np.ndarray:
    """Convenience function for single file preprocessing"""
    preprocessor = SharedPreprocessor(config)
    return preprocessor.load_and_preprocess_image(image_path)