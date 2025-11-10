"""
End-to-End Bite Mark Detection & Classification Pipeline
Three-stage system: Detection → Segmentation → Classification
Target: >99% accuracy with robust performance
"""

import numpy as np
import cv2
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class BiteMarkPrediction:
    """Complete prediction for a single bite mark instance."""
    # Detection
    bbox: List[int]  # [x1, y1, x2, y2]
    detection_confidence: float
    
    # Segmentation
    mask: np.ndarray  # Binary mask (H, W)
    segmentation_confidence: float
    mask_area: int
    mask_perimeter: float
    
    # Classification
    class_name: str
    class_id: int
    class_probabilities: Dict[str, float]
    classification_confidence: float
    
    # Overall
    overall_confidence: float  # Combined confidence score
    
    # Metadata
    bbox_crop: Optional[np.ndarray] = None
    attention_map: Optional[np.ndarray] = None


class BiteMarkDetectionPipeline:
    """
    Complete three-stage pipeline for bite mark detection and classification.
    
    Architecture:
        Stage 1: YOLOv9-E for detection
        Stage 2: SAM2 for segmentation
        Stage 3: ViT + EfficientNetV2 for classification
    """
    
    def __init__(
        self,
        detector_path: str = 'models/yolov9_bitemark.pt',
        segmenter_path: str = 'models/sam2_hiera_l.pt',
        classifier_path: str = 'models/classifier_ensemble.pt',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        confidence_threshold: float = 0.5
    ):
        """
        Initialize all three pipeline stages.
        
        Args:
            detector_path: YOLOv9 model path
            segmenter_path: SAM2 model path
            classifier_path: ViT+EfficientNet classifier path
            device: 'cuda' or 'cpu'
            confidence_threshold: Minimum confidence for detections
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.class_names = ['human', 'dog', 'snake']
        
        print("🚀 Initializing Bite Mark Detection Pipeline...")
        
        # Stage 1: Load YOLOv9 detector
        print("  [1/3] Loading YOLOv9 detector...")
        from detection_yolov9 import BiteMarkDetector
        self.detector = BiteMarkDetector(
            model_path=detector_path,
            confidence_threshold=confidence_threshold,
            device=device
        )
        
        # Stage 2: Load SAM2 segmenter
        print("  [2/3] Loading SAM2 segmenter...")
        from segmentation_sam2 import SAM2BiteMarkSegmenter
        self.segmenter = SAM2BiteMarkSegmenter(
            model_path=segmenter_path,
            device=device
        )
        
        # Stage 3: Load ViT+EfficientNet classifier
        print("  [3/3] Loading ViT+EfficientNet classifier...")
        from classification_vit import BiteMarkClassifier
        self.classifier = BiteMarkClassifier(
            num_classes=len(self.class_names),
            pretrained=False  # Load trained weights
        )
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        self.classifier.to(device)
        self.classifier.eval()
        
        # Load preprocessor
        from preprocessing_advanced import ForensicImagePreprocessor
        self.preprocessor = ForensicImagePreprocessor(target_size=(640, 640))
        
        print("✅ Pipeline initialization complete!\n")
    
    def predict(
        self, 
        image: np.ndarray,
        return_crops: bool = True,
        return_attention: bool = True
    ) -> List[BiteMarkPrediction]:
        """
        Run complete pipeline on single image.
        
        Args:
            image: Input RGB image (H, W, 3)
            return_crops: Return cropped bite mark regions
            return_attention: Generate Grad-CAM attention maps
            
        Returns:
            List of BiteMarkPrediction objects
        """
        predictions = []
        
        # Stage 1: Detect bite marks
        detections = self.detector.detect(image, return_crops=return_crops)
        
        if len(detections) == 0:
            return predictions  # No detections
        
        # Process each detection
        for det in detections:
            bbox = det['bbox']
            detection_conf = det['confidence']
            
            # Stage 2: Segment bite mark
            seg_result = self.segmenter.segment_from_bbox(image, bbox)
            
            # Extract cropped region for classification
            x1, y1, x2, y2 = bbox
            crop = image[y1:y2, x1:x2]
            
            # Preprocess crop for classifier
            crop_resized = cv2.resize(crop, (224, 224))
            crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
            crop_tensor = crop_tensor.unsqueeze(0).to(self.device)
            
            # Stage 3: Classify bite mark type
            with torch.no_grad():
                logits, attn_weights = self.classifier(crop_tensor)
                probs = torch.softmax(logits, dim=1)
                class_id = torch.argmax(probs, dim=1).item()
                class_conf = probs[0, class_id].item()
            
            # Build class probability dict
            class_probs = {
                self.class_names[i]: probs[0, i].item() 
                for i in range(len(self.class_names))
            }
            
            # Calculate overall confidence (geometric mean)
            overall_conf = (detection_conf * seg_result.confidence * class_conf) ** (1/3)
            
            # Optional: Generate attention map
            attention_map = None
            if return_attention:
                from classification_vit import GradCAMVisualizer
                target_layer = self.classifier.vit.blocks[-1]
                grad_cam = GradCAMVisualizer(self.classifier, target_layer)
                attention_map = grad_cam.generate_cam(crop_tensor, target_class=class_id)
            
            # Create prediction object
            prediction = BiteMarkPrediction(
                bbox=bbox,
                detection_confidence=detection_conf,
                mask=seg_result.mask,
                segmentation_confidence=seg_result.confidence,
                mask_area=seg_result.area,
                mask_perimeter=seg_result.perimeter,
                class_name=self.class_names[class_id],
                class_id=class_id,
                class_probabilities=class_probs,
                classification_confidence=class_conf,
                overall_confidence=overall_conf,
                bbox_crop=crop if return_crops else None,
                attention_map=attention_map
            )
            
            predictions.append(prediction)
        
        # Sort by overall confidence
        predictions = sorted(predictions, key=lambda x: x.overall_confidence, reverse=True)
        
        return predictions
    
    def predict_batch(
        self, 
        images: List[np.ndarray]
    ) -> List[List[BiteMarkPrediction]]:
        """
        Run pipeline on batch of images.
        
        Args:
            images: List of RGB images
            
        Returns:
            List of prediction lists (one per image)
        """
        all_predictions = []
        
        for image in images:
            predictions = self.predict(image)
            all_predictions.append(predictions)
        
        return all_predictions
    
    def visualize(
        self, 
        image: np.ndarray, 
        predictions: List[BiteMarkPrediction],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Comprehensive visualization of all predictions.
        
        Args:
            image: Original image
            predictions: List of BiteMarkPrediction
            save_path: Optional path to save
            
        Returns:
            Annotated image
        """
        viz = image.copy()
        
        # Color map
        colors = {
            'human': (255, 0, 0),
            'dog': (0, 255, 0),
            'snake': (0, 0, 255)
        }
        
        for pred in predictions:
            color = colors.get(pred.class_name, (255, 255, 0))
            
            # Draw segmentation mask
            mask_colored = np.zeros_like(viz)
            mask_colored[pred.mask > 0] = color
            viz = cv2.addWeighted(viz, 0.7, mask_colored, 0.3, 0)
            
            # Draw bounding box
            x1, y1, x2, y2 = pred.bbox
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_lines = [
                f"{pred.class_name.upper()}",
                f"Conf: {pred.overall_confidence:.1%}",
                f"Area: {pred.mask_area}px"
            ]
            
            y_offset = y1 - 10
            for line in reversed(label_lines):
                label_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(
                    viz,
                    (x1, y_offset - label_size[1] - 5),
                    (x1 + label_size[0] + 5, y_offset),
                    color,
                    -1
                )
                cv2.putText(
                    viz,
                    line,
                    (x1 + 2, y_offset - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                y_offset -= label_size[1] + 7
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        
        return viz
    
    def export_results(
        self, 
        predictions: List[BiteMarkPrediction],
        output_path: str
    ):
        """
        Export predictions to JSON format.
        
        Args:
            predictions: List of predictions
            output_path: Path to save JSON file
        """
        results = []
        
        for i, pred in enumerate(predictions):
            result = {
                'detection_id': i + 1,
                'bbox': pred.bbox,
                'class': pred.class_name,
                'class_probabilities': pred.class_probabilities,
                'confidence': {
                    'detection': float(pred.detection_confidence),
                    'segmentation': float(pred.segmentation_confidence),
                    'classification': float(pred.classification_confidence),
                    'overall': float(pred.overall_confidence)
                },
                'mask_metrics': {
                    'area_pixels': int(pred.mask_area),
                    'perimeter_pixels': float(pred.mask_perimeter)
                }
            }
            results.append(result)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results exported to {output_path}")


# Evaluation metrics
class DetectionMetrics:
    """Calculate detection and segmentation metrics."""
    
    @staticmethod
    def calculate_iou(box1: List[int], box2: List[int]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def calculate_map(
        predictions: List[Dict],
        ground_truths: List[Dict],
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate mAP (mean Average Precision).
        
        Args:
            predictions: List of {bbox, confidence, class_id}
            ground_truths: List of {bbox, class_id}
            iou_threshold: IoU threshold for TP
            
        Returns:
            Dict with mAP metrics
        """
        # Sort predictions by confidence
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        tp = 0
        fp = 0
        fn = len(ground_truths)
        
        matched_gt = set()
        
        for pred in predictions:
            best_iou = 0.0
            best_gt_idx = -1
            
            for i, gt in enumerate(ground_truths):
                if i in matched_gt:
                    continue
                
                if pred['class_id'] != gt['class_id']:
                    continue
                
                iou = DetectionMetrics.calculate_iou(pred['bbox'], gt['bbox'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp += 1
                fn -= 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }


# Example usage
if __name__ == '__main__':
    # Initialize pipeline
    pipeline = BiteMarkDetectionPipeline(
        detector_path='models/yolov9_bitemark.pt',
        segmenter_path='models/sam2_hiera_l.pt',
        classifier_path='models/classifier_ensemble.pt',
        device='cuda',
        confidence_threshold=0.5
    )
    
    # Load test image
    image = cv2.imread('data/test/sample.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run pipeline
    print("🔍 Running detection pipeline...")
    predictions = pipeline.predict(image, return_crops=True, return_attention=True)
    
    # Print results
    print(f"\n✅ Detected {len(predictions)} bite mark(s):\n")
    for i, pred in enumerate(predictions):
        print(f"[{i+1}] {pred.class_name.upper()}")
        print(f"  Overall Confidence: {pred.overall_confidence:.2%}")
        print(f"  Detection: {pred.detection_confidence:.2%}")
        print(f"  Segmentation: {pred.segmentation_confidence:.2%}")
        print(f"  Classification: {pred.classification_confidence:.2%}")
        print(f"  Mask Area: {pred.mask_area} pixels")
        print(f"  Class Probabilities:")
        for cls, prob in pred.class_probabilities.items():
            print(f"    {cls}: {prob:.2%}")
        print()
    
    # Visualize
    viz = pipeline.visualize(image, predictions, save_path='outputs/pipeline_result.jpg')
    print("✅ Visualization saved to outputs/pipeline_result.jpg")
    
    # Export results
    pipeline.export_results(predictions, 'outputs/predictions.json')
