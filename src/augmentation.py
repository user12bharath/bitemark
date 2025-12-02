"""
Advanced Data Augmentation Module for Bite Mark Classification
Preserves bite mark integrity while increasing dataset diversity
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from scipy.ndimage import rotate, zoom

# Global seed for deterministic augmentation
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


class AugmentationConfig:
    """Configuration class for augmentation parameters"""
    def __init__(self):
        self.brightness = (0.7, 1.3)  # Wider range for forensic photos
        self.contrast = (0.8, 1.2)
        self.noise_level = 0.01  # Reduced noise
        self.blur_kernel = 1  # Minimal blur
        self.rotation_range = 15  # Degrees
        self.zoom_range = (0.95, 1.05)
        self.shear_range = (-0.1, 0.1)
        self.perspective_strength = 0.05  # Reduced perspective
        self.saturation = (0.8, 1.2)
        self.disable_vertical_flip = False  # Can be set True for human bites
        
        # Probabilities for each augmentation
        self.prob_rotation = 0.6
        self.prob_flip_h = 0.5
        self.prob_flip_v = 0.3
        self.prob_brightness = 0.7
        self.prob_contrast = 0.7
        self.prob_saturation = 0.5
        self.prob_noise = 0.4
        self.prob_blur = 0.3
        self.prob_zoom = 0.4
        self.prob_shear = 0.3
        self.prob_perspective = 0.1  # Reduced from 0.2 to avoid errors


class BiteMarkAugmentor:
    """
    Advanced augmentation techniques specifically designed for bite mark images
    Optimized for real forensic images with class imbalance handling
    """
    
    def __init__(self, preserve_features=True, balance_classes=True, config=None):
        """
        Initialize augmentor
        
        Args:
            preserve_features: If True, limit augmentation to preserve bite patterns
            balance_classes: If True, augment minority classes more to balance dataset
            config: AugmentationConfig object for parameter control
        """
        self.preserve_features = preserve_features
        self.balance_classes = balance_classes
        self.config = config if config else AugmentationConfig()
        
    def augment_dataset(self, images, labels, augmentation_factor=2, 
                       class_names=None, save_augmented=False, augmented_dir='data/augmented'):
        """
        Augment entire dataset with class balancing
        
        Args:
            images: NumPy array of images
            labels: NumPy array of labels
            augmentation_factor: Base augmentation factor
            class_names: List of class names for balancing
            save_augmented: If True, save augmented images to disk
            augmented_dir: Directory to save augmented images
            
        Returns:
            Augmented images and labels
        """
        print(f"\n🔄 Augmenting dataset with class balancing...")
        
        # Create augmented directory if saving
        if save_augmented and class_names is not None:
            os.makedirs(augmented_dir, exist_ok=True)
            for class_name in class_names:
                os.makedirs(os.path.join(augmented_dir, class_name), exist_ok=True)
        
        if self.balance_classes and class_names is not None:
            # Calculate class-specific augmentation factors
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            max_count = label_counts.max()
            
            print(f"\n  Original class distribution:")
            for label, count in zip(unique_labels, label_counts):
                class_name = class_names[label] if label < len(class_names) else f"Class_{label}"
                print(f"    {class_name}: {count} images")
            
            augmented_images = []
            augmented_labels = []
            
            for label in unique_labels:
                # Get images for this class
                class_mask = labels == label
                class_images = images[class_mask]
                class_labels = labels[class_mask]
                
                # DETERMINISTIC OVERSAMPLING: Calculate augmentation factor for this class
                # Target: balance all classes to max_count * augmentation_factor
                current_count = len(class_images)
                target_count = max_count * augmentation_factor
                class_aug_factor = max(1, int(np.ceil(target_count / current_count)))
                
                class_name = class_names[label] if label < len(class_names) else f"Class_{label}"
                print(f"\n  Augmenting {class_name}: {current_count} → {current_count + (current_count * (class_aug_factor - 1))} images (factor={class_aug_factor}x)")
                
                # Add original images
                augmented_images.append(class_images)
                augmented_labels.append(class_labels)
                
                # DETERMINISTIC: Set seed for each augmentation round
                # Generate augmented versions
                for i in range(class_aug_factor - 1):
                    np.random.seed(SEED + i)  # Deterministic per round
                    aug_imgs = []
                    for idx, img in enumerate(class_images):
                        aug_img = self.apply_random_augmentation(img)
                        
                        # Ensure consistent shape and type
                        if hasattr(aug_img, 'numpy'):  # TensorFlow tensor
                            aug_img = aug_img.numpy()
                        
                        # Ensure correct shape (match original image)
                        original_shape = img.shape
                        if aug_img.shape != original_shape:
                            # Resize to match original if needed
                            if len(original_shape) == 3:
                                aug_img = cv2.resize(aug_img, (original_shape[1], original_shape[0]))
                                if len(aug_img.shape) == 2:  # Add channel dimension if lost
                                    aug_img = aug_img[..., np.newaxis]
                            else:
                                aug_img = cv2.resize(aug_img, (original_shape[1], original_shape[0]))
                        
                        # Ensure float32 type for consistency
                        aug_img = aug_img.astype(np.float32)
                        
                        aug_imgs.append(aug_img)
                        
                        # Save augmented image if requested
                        if save_augmented:
                            filename = f"{class_name}_aug{i+1}_{idx:04d}.jpg"
                            output_path = os.path.join(augmented_dir, class_name, filename)
                            # Convert to uint8 for saving
                            save_img = (np.clip(aug_img, 0, 1) * 255).astype(np.uint8)
                            cv2.imwrite(output_path, save_img)
                    
                    # Ensure all images have exactly the same shape before creating array
                    if aug_imgs:
                        expected_shape = aug_imgs[0].shape
                        for j, aug_img in enumerate(aug_imgs):
                            if aug_img.shape != expected_shape:
                                print(f"    ⚠ Shape mismatch fixed for image {j}: {aug_img.shape} -> {expected_shape}")
                                if len(expected_shape) == 3:
                                    aug_imgs[j] = cv2.resize(aug_img, (expected_shape[1], expected_shape[0]))
                                    if len(aug_imgs[j].shape) == 2:
                                        aug_imgs[j] = aug_imgs[j][..., np.newaxis]
                                else:
                                    aug_imgs[j] = cv2.resize(aug_img, (expected_shape[1], expected_shape[0]))
                                aug_imgs[j] = aug_imgs[j].astype(np.float32)
                    
                    augmented_images.append(np.array(aug_imgs))
                    augmented_labels.append(class_labels)
                    print(f"    ✓ Created augmentation set {i+1}/{class_aug_factor-1}")
            
            final_images = np.concatenate(augmented_images, axis=0)
            final_labels = np.concatenate(augmented_labels, axis=0)
            
        else:
            # Standard augmentation without balancing
            augmented_images = [images]
            augmented_labels = [labels]
            
            for i in range(augmentation_factor - 1):
                np.random.seed(SEED + i)  # Deterministic
                aug_imgs = []
                for idx, (img, label) in enumerate(zip(images, labels)):
                    aug_img = self.apply_random_augmentation(img)
                    aug_imgs.append(aug_img)
                    
                    # Save augmented image if requested
                    if save_augmented and class_names is not None:
                        class_name = class_names[label] if label < len(class_names) else f"class_{label}"
                        filename = f"{class_name}_aug{i+1}_{idx:04d}.jpg"
                        output_path = os.path.join(augmented_dir, class_name, filename)
                        save_img = (np.clip(aug_img, 0, 1) * 255).astype(np.uint8)
                        cv2.imwrite(output_path, save_img)
                
                augmented_images.append(np.array(aug_imgs))
                augmented_labels.append(labels)
                print(f"  ✓ Created augmentation set {i+1}/{augmentation_factor-1}")
            
            final_images = np.concatenate(augmented_images, axis=0)
            final_labels = np.concatenate(augmented_labels, axis=0)
        
        if save_augmented:
            print(f"\n💾 Saved {len(final_images) - len(images)} augmented images to {augmented_dir}")
        
        print(f"\n✓ Augmentation complete: {len(images)} → {len(final_images)} samples")
        
        # Print final distribution
        if class_names is not None:
            unique_labels, label_counts = np.unique(final_labels, return_counts=True)
            print(f"\n  Final class distribution:")
            for label, count in zip(unique_labels, label_counts):
                class_name = class_names[label] if label < len(class_names) else f"Class_{label}"
                percentage = (count / len(final_labels)) * 100
                print(f"    {class_name}: {count} images ({percentage:.1f}%)")
        
        return final_images, final_labels
    
    def apply_random_augmentation(self, image):
        """Apply random combination of augmentations to single image - optimized for real photos"""
        img = image.copy()
        
        # Random rotation (limited to preserve orientation)
        if np.random.random() < self.config.prob_rotation:
            img = self._rotate(img, angle_range=(-self.config.rotation_range, self.config.rotation_range))
        
        # Random flip (horizontal only by default)
        if np.random.random() < self.config.prob_flip_h:
            img = self._flip_horizontal(img)
        
        # Random vertical flip (can be disabled for human bites)
        if not self.config.disable_vertical_flip and np.random.random() < self.config.prob_flip_v:
            img = tf.image.flip_up_down(img).numpy()
        
        # Random brightness adjustment (important for real photos with varying lighting)
        if np.random.random() < self.config.prob_brightness:
            img = self._adjust_brightness(img, factor_range=self.config.brightness)
        
        # Random contrast adjustment
        if np.random.random() < self.config.prob_contrast:
            img = self._adjust_contrast(img, factor_range=self.config.contrast)
        
        # Random saturation (for color images)
        if (np.random.random() < self.config.prob_saturation and 
            len(img.shape) == 3 and img.shape[2] == 3):
            img = self._adjust_saturation(img, factor_range=self.config.saturation)
        
        # Random noise (subtle, to simulate camera noise)
        if np.random.random() < self.config.prob_noise:
            img = self._add_noise(img, noise_level=self.config.noise_level)
        
        # Random blur (very subtle, to simulate focus variation)
        if np.random.random() < self.config.prob_blur:
            img = self._apply_blur(img, kernel_size=self.config.blur_kernel)
        
        # Random zoom (slight)
        if np.random.random() < self.config.prob_zoom and not self.preserve_features:
            img = self._zoom(img, zoom_range=self.config.zoom_range)
        
        # Random shear (slight, to simulate angle variation)
        if np.random.random() < self.config.prob_shear:
            img = self._shear(img, shear_range=self.config.shear_range)
        
        # Random perspective transform (very subtle)
        if np.random.random() < self.config.prob_perspective:
            img = self._perspective_transform(img, strength=self.config.perspective_strength)
        
        # Ensure consistent output format
        if hasattr(img, 'numpy'):  # Convert TensorFlow tensor to numpy
            img = img.numpy()
        
        # Ensure float32 type
        img = img.astype(np.float32)
        
        # Ensure values are in [0, 1] range
        img = np.clip(img, 0.0, 1.0)
        
        return img
        
        # Random vertical flip (for bite marks, this can be valid)
        if np.random.random() > 0.7:
            img = np.flipud(img)
        
        # Random brightness adjustment (important for real photos with varying lighting)
        if np.random.random() > 0.3:
            img = self._adjust_brightness(img, factor_range=(0.7, 1.3))
        
        # Random contrast adjustment
        if np.random.random() > 0.3:
            img = self._adjust_contrast(img, factor_range=(0.7, 1.3))
        
        # Random saturation (for color images)
        if np.random.random() > 0.5 and len(img.shape) == 3 and img.shape[2] == 3:
            img = self._adjust_saturation(img, factor_range=(0.8, 1.2))
        
        # Random noise (subtle, to simulate camera noise)
        if np.random.random() > 0.6:
            img = self._add_noise(img, noise_level=0.015)
        
        # Random blur (very subtle, to simulate focus variation)
        if np.random.random() > 0.7:
            img = self._apply_blur(img, kernel_size=3)
        
        # Random zoom (slight)
        if np.random.random() > 0.6 and not self.preserve_features:
            img = self._zoom(img, zoom_range=(0.9, 1.1))
        
        # Random shear (slight, to simulate angle variation)
        if np.random.random() > 0.7:
            img = self._shear(img, shear_range=(-0.1, 0.1))
        
        # Random perspective transform (very subtle)
        if np.random.random() > 0.8 and not self.preserve_features:
            img = self._perspective_transform(img)
        
        return np.clip(img, 0, 1)
    
    def _rotate(self, image, angle_range=(-15, 15)):
        """Rotate image by random angle within range"""
        angle = np.random.uniform(*angle_range)
        
        if len(image.shape) == 3:
            rotated = rotate(image, angle, reshape=False, mode='nearest')
        else:
            rotated = rotate(image, angle, reshape=False, mode='nearest')
        
        return rotated
    
    def _flip_horizontal(self, image):
        """Flip image horizontally"""
        return np.fliplr(image)
    
    def _adjust_brightness(self, image, factor_range=(0.8, 1.2)):
        """Adjust image brightness"""
        factor = np.random.uniform(*factor_range)
        return image * factor
    
    def _adjust_contrast(self, image, factor_range=(0.8, 1.2)):
        """Adjust image contrast"""
        factor = np.random.uniform(*factor_range)
        mean = np.mean(image)
        return (image - mean) * factor + mean
    
    def _add_noise(self, image, noise_level=0.02):
        """Add Gaussian noise to image"""
        noise = np.random.normal(0, noise_level, image.shape)
        return image + noise
    
    def _apply_blur(self, image, kernel_size=3):
        """Apply Gaussian blur"""
        if len(image.shape) == 3 and image.shape[2] == 1:
            # Handle single channel
            blurred = cv2.GaussianBlur(image[:, :, 0], (kernel_size, kernel_size), 0)
            return np.expand_dims(blurred, axis=-1)
        else:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def _zoom(self, image, zoom_range=(0.95, 1.05)):
        """Apply random zoom"""
        zoom_factor = np.random.uniform(*zoom_range)
        
        if len(image.shape) == 3:
            zoomed = zoom(image, (zoom_factor, zoom_factor, 1), mode='nearest')
        else:
            zoomed = zoom(image, zoom_factor, mode='nearest')
        
        # Crop or pad to original size
        h, w = image.shape[:2]
        zh, zw = zoomed.shape[:2]
        
        if zoom_factor > 1:
            # Crop center
            start_h = (zh - h) // 2
            start_w = (zw - w) // 2
            result = zoomed[start_h:start_h+h, start_w:start_w+w]
        else:
            # Pad
            pad_h = (h - zh) // 2
            pad_w = (w - zw) // 2
            if len(image.shape) == 3:
                result = np.pad(zoomed, 
                              ((pad_h, h-zh-pad_h), (pad_w, w-zw-pad_w), (0, 0)),
                              mode='edge')
            else:
                result = np.pad(zoomed,
                              ((pad_h, h-zh-pad_h), (pad_w, w-zw-pad_w)),
                              mode='edge')
        
        return result
    
    def _adjust_saturation(self, image, factor_range=(0.8, 1.2)):
        """Adjust color saturation (for color images)"""
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image
        
        factor = np.random.uniform(*factor_range)
        
        # Convert to HSV
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        
        # Convert back to BGR
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
        return result
    
    def _shear(self, image, shear_range=(-0.1, 0.1)):
        """Apply random shear transformation"""
        shear_factor = np.random.uniform(*shear_range)
        h, w = image.shape[:2]
        
        # Create shear matrix
        M = np.array([[1, shear_factor, 0],
                     [0, 1, 0]], dtype=np.float32)
        
        sheared = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return sheared
    
    def _perspective_transform(self, image, strength=0.1):
        """Apply perspective transformation with proper error handling"""
        try:
            if len(image.shape) == 2:
                h, w = image.shape
            else:
                h, w = image.shape[:2]
            
            # Skip if image is too small
            if h < 10 or w < 10:
                return image
            
            # Define source points (corners) - ensure proper format
            src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
            
            # Define destination points with controlled perturbation
            max_offset = min(h, w) * strength * 0.1  # Much smaller offset
            dst_points = src_points.copy()
            
            # Add small random perturbations
            for i in range(4):
                dst_points[i] += np.random.uniform(-max_offset, max_offset, 2)
            
            # Ensure points are within image bounds
            dst_points[:, 0] = np.clip(dst_points[:, 0], 0, w-1)
            dst_points[:, 1] = np.clip(dst_points[:, 1], 0, h-1)
            
            # Get perspective transform matrix
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply transformation
            transformed = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            return transformed.astype(np.float32)
        except Exception as e:
            # If perspective transform fails, return original image
            print(f"Warning: Perspective transform failed: {e}")
            return image
    
    def create_augmentation_pipeline(self):
        """
        Create TensorFlow data augmentation layer
        For use in model training pipeline
        """
        data_augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),  # ±10% rotation
            keras.layers.RandomContrast(0.2),
            keras.layers.RandomBrightness(0.2),
        ], name="data_augmentation")
        
        return data_augmentation


def main():
    """Test augmentation module"""
    print("🔧 Testing Data Augmentation Module...")
    
    # Create sample image
    sample_image = np.random.rand(224, 224, 1).astype(np.float32)
    
    # Test with custom config
    config = AugmentationConfig()
    config.disable_vertical_flip = True  # Example: disable for human bites
    
    augmentor = BiteMarkAugmentor(preserve_features=True, config=config)
    
    # Test single augmentation
    augmented = augmentor.apply_random_augmentation(sample_image)
    print(f"✓ Single augmentation: {sample_image.shape} → {augmented.shape}")
    
    # Test TensorFlow pipeline
    tf_augment_fn = augmentor.get_tf_augmentation_function()
    tf_sample = tf.convert_to_tensor(sample_image)
    tf_label = tf.convert_to_tensor([0])
    tf_augmented, _ = tf_augment_fn(tf_sample, tf_label)
    print(f"✓ TensorFlow augmentation: {tf_sample.shape} → {tf_augmented.shape}")
    
    # Test batch augmentation
    sample_images = np.random.rand(10, 224, 224, 1).astype(np.float32)
    sample_labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    
    aug_images, aug_labels = augmentor.augment_dataset(
        sample_images, sample_labels, augmentation_factor=3
    )
    
    print(f"✓ Batch augmentation: {len(sample_images)} → {len(aug_images)} samples")


if __name__ == "__main__":
    main()
