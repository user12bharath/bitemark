"""
Advanced Preprocessing for Forensic Bite Mark Detection
Implements state-of-the-art medical image enhancement techniques
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import exposure, filters, morphology
from typing import Tuple, Optional
import albumentations as A


class ForensicImagePreprocessor:
    """
    Advanced preprocessing for bite mark detection optimized for forensic images.
    Handles challenging conditions: poor lighting, low contrast, skin tone variations.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        Args:
            target_size: Target image dimensions (width, height)
        """
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
    def preprocess_image(self, image: np.ndarray, enhance: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline for single image.
        
        Args:
            image: Input image (H, W, C) in BGR or RGB
            enhance: Apply contrast enhancement
            
        Returns:
            Preprocessed image ready for model input
        """
        # Step 1: Color space correction
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Step 2: Illumination correction (Retinex)
        if enhance:
            image = self.apply_multi_scale_retinex(image)
        
        # Step 3: CLAHE in LAB color space
        image = self.apply_clahe_lab(image)
        
        # Step 4: Bilateral filtering (preserve edges, reduce noise)
        image = cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)
        
        # Step 5: Resize with aspect ratio preservation
        image = self.resize_with_padding(image, self.target_size)
        
        # Step 6: Normalization
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def apply_multi_scale_retinex(self, image: np.ndarray, 
                                   scales: list = [15, 80, 250]) -> np.ndarray:
        """
        Multi-Scale Retinex (MSR) for illumination correction.
        Critical for handling varying lighting in forensic photos.
        
        Args:
            image: Input RGB image
            scales: Gaussian scales for multi-scale processing
            
        Returns:
            Illumination-corrected image
        """
        image_float = image.astype(np.float32) + 1.0  # Avoid log(0)
        
        msr = np.zeros_like(image_float)
        for scale in scales:
            gaussian = cv2.GaussianBlur(image_float, (0, 0), scale)
            msr += np.log(image_float) - np.log(gaussian + 1.0)
        
        msr = msr / len(scales)
        
        # Normalize to [0, 255]
        msr = self._normalize_msr(msr)
        
        return msr.astype(np.uint8)
    
    def _normalize_msr(self, msr: np.ndarray) -> np.ndarray:
        """Normalize MSR output to valid image range."""
        for i in range(msr.shape[2]):
            channel = msr[:, :, i]
            min_val, max_val = np.min(channel), np.max(channel)
            if max_val > min_val:
                msr[:, :, i] = 255 * (channel - min_val) / (max_val - min_val)
        return msr
    
    def apply_clahe_lab(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB space.
        Enhances local contrast while preserving color information.
        
        Args:
            image: Input RGB image
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def resize_with_padding(self, image: np.ndarray, 
                           target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image while preserving aspect ratio using padding.
        
        Args:
            image: Input image
            target_size: (width, height)
            
        Returns:
            Resized and padded image
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create padded canvas
        canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
        
        # Calculate padding offsets (center the image)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Place resized image on canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def extract_skin_region(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract skin regions using YCrCb color space.
        Useful for filtering non-skin areas in forensic photos.
        
        Args:
            image: Input RGB image
            
        Returns:
            Tuple of (skin_mask, masked_image)
        """
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        # Skin color thresholds (adaptive for various skin tones)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create binary mask
        mask = cv2.inRange(ycrcb, lower, upper)
        
        # Morphological operations to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to image
        masked = cv2.bitwise_and(image, image, mask=mask)
        
        return mask, masked
    
    def enhance_bite_mark_features(self, image: np.ndarray) -> np.ndarray:
        """
        Domain-specific enhancement to highlight bite mark patterns.
        Uses edge detection and morphological operations.
        
        Args:
            image: Preprocessed RGB image
            
        Returns:
            Feature-enhanced image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Sobel edge detection (horizontal + vertical)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.hypot(sobelx, sobely)
        edges = (edges / edges.max() * 255).astype(np.uint8)
        
        # Combine with original
        enhanced = cv2.addWeighted(gray, 0.7, edges, 0.3, 0)
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return enhanced_rgb


class BiteMarkAugmentationPipeline:
    """
    Advanced augmentation pipeline specifically designed for bite mark detection.
    Uses Albumentations for efficient GPU-accelerated augmentation.
    """
    
    def __init__(self, mode: str = 'detection'):
        """
        Args:
            mode: 'detection', 'segmentation', or 'classification'
        """
        self.mode = mode
        self.train_transform = self._build_train_augmentation()
        self.val_transform = self._build_val_augmentation()
        
    def _build_train_augmentation(self) -> A.Compose:
        """
        Build training augmentation pipeline.
        Handles rotation, blur, lighting, elastic deformation.
        """
        transforms = [
            # Geometric augmentations
            A.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=45, 
                p=0.7,
                border_mode=cv2.BORDER_CONSTANT
            ),
            
            # Elastic deformation (crucial for bite mark variations)
            A.ElasticTransform(
                alpha=120, 
                sigma=120 * 0.05, 
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3),
            
            # Lighting and color augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.3, 
                contrast_limit=0.3, 
                p=0.7
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=20, 
                sat_shift_limit=30, 
                val_shift_limit=20, 
                p=0.5
            ),
            A.CLAHE(clip_limit=4.0, p=0.3),
            
            # Noise and blur (simulate poor image quality)
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(p=1.0),
                A.MultiplicativeNoise(p=1.0),
            ], p=0.5),
            
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
            ], p=0.4),
            
            # Simulate compression artifacts
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
            
            # Cutout / CoarseDropout (occlusion simulation)
            A.CoarseDropout(
                max_holes=8, 
                max_height=32, 
                max_width=32, 
                p=0.3
            ),
            
            # Normalize
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        
        # Add bounding box support for detection mode
        if self.mode == 'detection':
            return A.Compose(
                transforms, 
                bbox_params=A.BboxParams(
                    format='yolo', 
                    label_fields=['class_labels']
                )
            )
        # Add mask support for segmentation mode
        elif self.mode == 'segmentation':
            return A.Compose(transforms, additional_targets={'mask': 'mask'})
        else:
            return A.Compose(transforms)
    
    def _build_val_augmentation(self) -> A.Compose:
        """Validation augmentation (minimal, only normalize)."""
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __call__(self, image: np.ndarray, 
                 bboxes: Optional[list] = None,
                 masks: Optional[np.ndarray] = None,
                 class_labels: Optional[list] = None,
                 is_training: bool = True) -> dict:
        """
        Apply augmentation pipeline.
        
        Args:
            image: Input image (H, W, C)
            bboxes: Bounding boxes in YOLO format [[x_center, y_center, w, h], ...]
            masks: Segmentation masks (H, W) or (H, W, C)
            class_labels: Class labels for each bbox
            is_training: Use training or validation augmentation
            
        Returns:
            Dict with augmented image, bboxes, masks
        """
        transform = self.train_transform if is_training else self.val_transform
        
        # Prepare augmentation input
        aug_input = {'image': image}
        
        if self.mode == 'detection' and bboxes is not None:
            aug_input['bboxes'] = bboxes
            aug_input['class_labels'] = class_labels
        elif self.mode == 'segmentation' and masks is not None:
            aug_input['mask'] = masks
        
        # Apply augmentation
        augmented = transform(**aug_input)
        
        return augmented


# Example usage
if __name__ == '__main__':
    # Test preprocessing
    preprocessor = ForensicImagePreprocessor(target_size=(640, 640))
    
    # Load sample image
    import cv2
    image = cv2.imread('data/raw/human/sample.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    processed = preprocessor.preprocess_image(image, enhance=True)
    
    print(f"✅ Preprocessed image shape: {processed.shape}")
    print(f"✅ Value range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Test augmentation
    aug_pipeline = BiteMarkAugmentationPipeline(mode='detection')
    
    # Example bounding box (YOLO format: x_center, y_center, width, height)
    bboxes = [[0.5, 0.5, 0.3, 0.4]]
    class_labels = [0]  # 0=human
    
    augmented = aug_pipeline(
        image=processed, 
        bboxes=bboxes, 
        class_labels=class_labels,
        is_training=True
    )
    
    print(f"✅ Augmented image shape: {augmented['image'].shape}")
    print(f"✅ Augmented bboxes: {augmented.get('bboxes', 'N/A')}")
