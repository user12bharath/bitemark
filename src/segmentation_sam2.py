"""
SAM2-Based Bite Mark Segmentation Module
Segment Anything Model 2 for pixel-perfect bite mark masks
"""

import numpy as np
import cv2
import torch
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class SegmentationResult:
    """Container for segmentation results."""
    mask: np.ndarray  # Binary mask (H, W)
    bbox: List[int]   # Bounding box [x1, y1, x2, y2]
    confidence: float
    area: int
    perimeter: float
    contours: List[np.ndarray]


class SAM2BiteMarkSegmenter:
    """
    SAM2-based bite mark segmenter with fine-tuning support.
    Provides pixel-perfect segmentation masks from detection bboxes.
    """
    
    def __init__(
        self,
        model_path: str = 'models/sam2_vit_h.pt',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model_path: Path to SAM2 checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.model = self._load_sam2_model(model_path)
        
    def _load_sam2_model(self, model_path: str):
        """
        Load SAM2 model from checkpoint.
        """
        try:
            # Option 1: Official SAM2 (meta-llama/segment-anything-2)
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Build SAM2 model
            sam2_checkpoint = model_path
            model_cfg = "sam2_hiera_l.yaml"  # or sam2_hiera_b+.yaml, sam2_hiera_s.yaml
            
            sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
            predictor = SAM2ImagePredictor(sam2_model)
            
            return predictor
            
        except ImportError:
            print("⚠️  SAM2 not installed. Install with:")
            print("    pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            raise
    
    def segment_from_bbox(
        self,
        image: np.ndarray,
        bbox: List[int],
        use_point_prompts: bool = True
    ) -> SegmentationResult:
        """
        Generate segmentation mask from bounding box prompt.
        
        Args:
            image: Input RGB image (H, W, 3)
            bbox: Bounding box [x1, y1, x2, y2]
            use_point_prompts: Whether to add center point as additional prompt
            
        Returns:
            SegmentationResult with mask and metadata
        """
        # Set image for SAM2
        self.model.set_image(image)
        
        # Convert bbox to SAM2 format
        x1, y1, x2, y2 = bbox
        box_prompt = np.array([[x1, y1, x2, y2]])
        
        # Optional: Add center point as positive prompt
        point_prompts = None
        point_labels = None
        
        if use_point_prompts:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            point_prompts = np.array([[center_x, center_y]])
            point_labels = np.array([1])  # 1 = foreground
        
        # Predict mask
        masks, scores, _ = self.model.predict(
            point_coords=point_prompts,
            point_labels=point_labels,
            box=box_prompt,
            multimask_output=False  # Single best mask
        )
        
        # Get best mask
        mask = masks[0]
        confidence = float(scores[0])
        
        # Refine mask with morphological operations
        mask_refined = self._refine_mask(mask)
        
        # Extract contours
        contours = self._extract_contours(mask_refined)
        
        # Calculate metrics
        area = int(np.sum(mask_refined))
        perimeter = 0.0
        if contours:
            perimeter = cv2.arcLength(contours[0], closed=True)
        
        return SegmentationResult(
            mask=mask_refined,
            bbox=bbox,
            confidence=confidence,
            area=area,
            perimeter=perimeter,
            contours=contours
        )
    
    def segment_batch(
        self,
        image: np.ndarray,
        bboxes: List[List[int]]
    ) -> List[SegmentationResult]:
        """
        Segment multiple bounding boxes in one image.
        
        Args:
            image: Input RGB image
            bboxes: List of bounding boxes
            
        Returns:
            List of SegmentationResult
        """
        results = []
        
        for bbox in bboxes:
            result = self.segment_from_bbox(image, bbox)
            results.append(result)
        
        return results
    
    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine segmentation mask using morphological operations.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Refined binary mask
        """
        # Convert to uint8
        mask = (mask * 255).astype(np.uint8)
        
        # Morphological closing (fill small holes)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Morphological opening (remove small noise)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # Fill holes using contour filling
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Keep only the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            mask_filled = np.zeros_like(mask)
            cv2.drawContours(mask_filled, [largest_contour], -1, 255, thickness=cv2.FILLED)
            mask = mask_filled
        
        # Convert back to binary
        mask = (mask > 127).astype(np.uint8)
        
        return mask
    
    def _extract_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Extract contours from binary mask.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            List of contours sorted by area (largest first)
        """
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Sort by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        return contours
    
    def visualize_segmentation(
        self,
        image: np.ndarray,
        result: SegmentationResult,
        alpha: float = 0.5,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Visualize segmentation result on image.
        
        Args:
            image: Original RGB image
            result: SegmentationResult from segment_from_bbox()
            alpha: Transparency of mask overlay
            color: RGB color for mask
            
        Returns:
            Visualized image with overlay
        """
        viz = image.copy()
        
        # Create colored mask
        mask_colored = np.zeros_like(image)
        mask_colored[result.mask > 0] = color
        
        # Blend with original image
        viz = cv2.addWeighted(viz, 1 - alpha, mask_colored, alpha, 0)
        
        # Draw contours
        if result.contours:
            cv2.drawContours(viz, result.contours, -1, color, 2)
        
        # Draw bounding box
        x1, y1, x2, y2 = result.bbox
        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
        
        # Add text info
        text = f"Area: {result.area} | Conf: {result.confidence:.2%}"
        cv2.putText(
            viz, 
            text, 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            color, 
            2
        )
        
        return viz
    
    def calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two masks.
        
        Args:
            mask1: Binary mask 1
            mask2: Binary mask 2
            
        Returns:
            IoU score [0, 1]
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def calculate_dice(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate Dice coefficient between two masks.
        
        Args:
            mask1: Binary mask 1
            mask2: Binary mask 2
            
        Returns:
            Dice score [0, 1]
        """
        intersection = np.logical_and(mask1, mask2).sum()
        total = mask1.sum() + mask2.sum()
        
        if total == 0:
            return 0.0
        
        return (2 * intersection) / total


class MaskRefinementNetwork(torch.nn.Module):
    """
    Lightweight U-Net for post-processing SAM2 masks.
    Trained specifically on bite mark dataset for domain adaptation.
    """
    
    def __init__(self, in_channels: int = 4, out_channels: int = 1):
        """
        Args:
            in_channels: RGB image (3) + SAM2 mask (1) = 4
            out_channels: Refined binary mask (1)
        """
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        
        # Bottleneck
        self.bottleneck = self._conv_block(128, 256)
        
        # Decoder
        self.dec3 = self._upconv_block(256, 128)
        self.dec2 = self._upconv_block(128, 64)
        self.dec1 = self._upconv_block(64, 32)
        
        # Output
        self.output = torch.nn.Conv2d(32, out_channels, kernel_size=1)
        
    def _conv_block(self, in_ch: int, out_ch: int) -> torch.nn.Sequential:
        """Convolutional block with BatchNorm and ReLU."""
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_ch: int, out_ch: int) -> torch.nn.Sequential:
        """Upsampling block with transpose convolution."""
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 4, H, W) - RGB + SAM2 mask
            
        Returns:
            Refined mask (B, 1, H, W)
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(torch.nn.functional.max_pool2d(e1, 2))
        e3 = self.enc3(torch.nn.functional.max_pool2d(e2, 2))
        
        # Bottleneck
        b = self.bottleneck(torch.nn.functional.max_pool2d(e3, 2))
        
        # Decoder with skip connections
        d3 = self.dec3(b) + e3
        d2 = self.dec2(d3) + e2
        d1 = self.dec1(d2) + e1
        
        # Output
        out = torch.sigmoid(self.output(d1))
        
        return out


# Example usage
if __name__ == '__main__':
    # Initialize segmenter
    segmenter = SAM2BiteMarkSegmenter(
        model_path='models/sam2_hiera_l.pt',
        device='cuda'
    )
    
    # Load image
    image = cv2.imread('data/test/sample.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Example bounding box from YOLOv9
    bbox = [100, 150, 300, 400]
    
    # Segment
    result = segmenter.segment_from_bbox(image, bbox)
    
    print(f"✅ Segmentation complete:")
    print(f"  Mask shape: {result.mask.shape}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Area: {result.area} pixels")
    print(f"  Perimeter: {result.perimeter:.1f} pixels")
    
    # Visualize
    viz = segmenter.visualize_segmentation(image, result)
    cv2.imwrite('outputs/segmentation_result.jpg', cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
    
    print("✅ Visualization saved to outputs/segmentation_result.jpg")
