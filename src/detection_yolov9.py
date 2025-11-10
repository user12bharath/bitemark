"""
YOLOv9-Based Bite Mark Detection Module
State-of-the-art object detection for bite mark localization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
import cv2


class BiteMarkDetector:
    """
    YOLOv9-based bite mark detector with custom post-processing.
    Handles detection, NMS, and confidence filtering.
    """
    
    def __init__(
        self,
        model_path: str = 'models/yolov9_bitemark.pt',
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model_path: Path to trained YOLOv9 weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: 'cuda' or 'cpu'
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Class names
        self.class_names = ['human', 'dog', 'snake']
        self.num_classes = len(self.class_names)
        
        # Load YOLOv9 model
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        """
        Load YOLOv9 model from checkpoint.
        Uses ultralytics YOLOv9 implementation.
        """
        try:
            # Option 1: Using ultralytics (recommended)
            from ultralytics import YOLO
            model = YOLO(model_path)
            model.to(self.device)
            return model
        except ImportError:
            # Option 2: Using torch.hub (YOLOv9 official)
            print("⚠️  ultralytics not found, using torch.hub...")
            model = torch.hub.load(
                'WongKinYiu/yolov9', 
                'custom', 
                path=model_path,
                force_reload=False
            )
            model.to(self.device)
            return model
    
    def detect(
        self, 
        image: np.ndarray,
        return_crops: bool = False
    ) -> List[Dict]:
        """
        Detect bite marks in image.
        
        Args:
            image: Input image (H, W, C) in RGB format
            return_crops: Whether to return cropped bite mark regions
            
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - class_id: int
                - class_name: str
                - crop: cropped image (if return_crops=True)
        """
        # Run inference
        results = self.model(image, conf=self.confidence_threshold, iou=self.iou_threshold)
        
        detections = []
        
        # Parse results
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Extract box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.class_names[class_id]
                }
                
                # Optionally crop bite mark region
                if return_crops:
                    crop = image[int(y1):int(y2), int(x1):int(x2)]
                    detection['crop'] = crop
                
                detections.append(detection)
        
        return detections
    
    def detect_batch(
        self, 
        images: List[np.ndarray]
    ) -> List[List[Dict]]:
        """
        Batch detection for multiple images.
        
        Args:
            images: List of images (H, W, C)
            
        Returns:
            List of detection lists (one per image)
        """
        all_detections = []
        
        for image in images:
            detections = self.detect(image)
            all_detections.append(detections)
        
        return all_detections
    
    def visualize_detections(
        self, 
        image: np.ndarray, 
        detections: List[Dict],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image: Input image
            detections: List of detections from detect()
            save_path: Optional path to save visualization
            
        Returns:
            Annotated image
        """
        viz = image.copy()
        
        # Color map for classes
        colors = {
            'human': (255, 0, 0),    # Red
            'dog': (0, 255, 0),      # Green
            'snake': (0, 0, 255)     # Blue
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            color = colors.get(class_name, (255, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                viz, 
                (x1, y1 - label_size[1] - 10), 
                (x1 + label_size[0], y1), 
                color, 
                -1
            )
            cv2.putText(
                viz, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        
        return viz


class YOLOv9Config:
    """
    Configuration for training YOLOv9 on bite mark dataset.
    """
    
    # Data configuration
    dataset_yaml = """
    # Bite Mark Detection Dataset
    path: ../data  # dataset root dir
    train: train/images  # train images
    val: val/images      # val images
    test: test/images    # test images (optional)
    
    # Classes
    names:
      0: human
      1: dog
      2: snake
    
    # Number of classes
    nc: 3
    """
    
    # Training hyperparameters
    hyperparameters = {
        # Model
        'model': 'yolov9-e.yaml',  # YOLOv9-E (largest, most accurate)
        'pretrained': 'yolov9-e.pt',  # Pre-trained COCO weights
        
        # Training
        'epochs': 300,
        'batch_size': 16,  # Adjust based on GPU memory
        'imgsz': 640,      # Input image size
        'device': 'cuda',  # or 'cpu'
        
        # Optimizer
        'optimizer': 'AdamW',
        'lr0': 0.001,      # Initial learning rate
        'lrf': 0.01,       # Final learning rate (lr0 * lrf)
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Scheduler
        'cos_lr': True,    # Cosine LR scheduler
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Loss weights
        'box': 7.5,        # Box loss weight
        'cls': 0.5,        # Classification loss weight
        'dfl': 1.5,        # Distribution Focal Loss weight
        
        # Augmentation
        'hsv_h': 0.015,    # HSV-Hue augmentation
        'hsv_s': 0.7,      # HSV-Saturation augmentation
        'hsv_v': 0.4,      # HSV-Value augmentation
        'degrees': 10.0,   # Rotation
        'translate': 0.1,  # Translation
        'scale': 0.9,      # Scale
        'shear': 0.0,      # Shear
        'perspective': 0.0, # Perspective
        'flipud': 0.5,     # Vertical flip
        'fliplr': 0.5,     # Horizontal flip
        'mosaic': 1.0,     # Mosaic augmentation
        'mixup': 0.15,     # Mixup augmentation
        'copy_paste': 0.3, # Copy-paste augmentation
        
        # Regularization
        'dropout': 0.0,    # Dropout (disabled for YOLOv9)
        'label_smoothing': 0.0,
        
        # Post-processing
        'conf': 0.5,       # Confidence threshold
        'iou': 0.45,       # NMS IoU threshold
        'max_det': 100,    # Maximum detections per image
        
        # Logging
        'save_period': 10,  # Save checkpoint every N epochs
        'plots': True,      # Save training plots
        'val': True,        # Validate during training
    }
    
    @staticmethod
    def get_training_command() -> str:
        """Generate YOLOv9 training command."""
        return """
# Train YOLOv9 on bite mark dataset
yolo detect train \\
    model=yolov9-e.yaml \\
    data=data/bitemark.yaml \\
    epochs=300 \\
    imgsz=640 \\
    batch=16 \\
    device=0 \\
    optimizer=AdamW \\
    lr0=0.001 \\
    cos_lr=True \\
    warmup_epochs=3 \\
    box=7.5 \\
    cls=0.5 \\
    dfl=1.5 \\
    mosaic=1.0 \\
    mixup=0.15 \\
    copy_paste=0.3 \\
    conf=0.5 \\
    iou=0.45 \\
    save=True \\
    plots=True \\
    val=True
"""


# Training script
def train_yolov9_bite_mark_detector(
    data_yaml: str = 'data/bitemark.yaml',
    epochs: int = 300,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = 'cuda',
    save_dir: str = 'runs/detect/bitemark'
):
    """
    Train YOLOv9 bite mark detector.
    
    Args:
        data_yaml: Path to dataset YAML configuration
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Input image size
        device: 'cuda' or 'cpu'
        save_dir: Directory to save results
    """
    from ultralytics import YOLO
    
    # Initialize model
    model = YOLO('yolov9-e.yaml')
    
    # Load pretrained weights
    model.load('yolov9-e.pt')
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        save=True,
        plots=True,
        val=True,
        project=save_dir
    )
    
    print(f"✅ Training complete! Results saved to {save_dir}")
    
    return results


# Example usage
if __name__ == '__main__':
    # Initialize detector
    detector = BiteMarkDetector(
        model_path='models/yolov9_bitemark.pt',
        confidence_threshold=0.5,
        iou_threshold=0.45
    )
    
    # Load test image
    image = cv2.imread('data/test/sample.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect bite marks
    detections = detector.detect(image, return_crops=True)
    
    # Print results
    print(f"✅ Detected {len(detections)} bite mark(s):")
    for i, det in enumerate(detections):
        print(f"  [{i+1}] {det['class_name']}: {det['confidence']:.2%} at {det['bbox']}")
    
    # Visualize
    viz = detector.visualize_detections(image, detections, save_path='outputs/detection_result.jpg')
    
    print("✅ Visualization saved to outputs/detection_result.jpg")
