"""
Vision Transformer (ViT) + EfficientNetV2 Ensemble Classifier
Fine-grained bite mark classification with attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2


class BiteMarkClassifier(nn.Module):
    """
    Ensemble classifier combining Vision Transformer and EfficientNetV2.
    Designed for fine-grained bite mark type classification.
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        vit_model: str = 'vit_base_patch16_224',
        efficientnet_model: str = 'efficientnetv2_m',
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Args:
            num_classes: Number of bite mark classes (human, dog, snake)
            vit_model: Vision Transformer variant
            efficientnet_model: EfficientNet variant
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load Vision Transformer (global context)
        self.vit = self._load_vit(vit_model, pretrained)
        vit_features = self.vit.head.in_features
        self.vit.head = nn.Identity()  # Remove original classifier
        
        # Load EfficientNetV2 (local texture details)
        self.efficientnet = self._load_efficientnet(efficientnet_model, pretrained)
        eff_features = self.efficientnet.classifier.in_features
        self.efficientnet.classifier = nn.Identity()
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(vit_features + eff_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Linear(256, num_classes)
        
        # Attention mechanism for feature weighting
        self.attention = nn.Sequential(
            nn.Linear(vit_features + eff_features, 2),
            nn.Softmax(dim=1)
        )
        
    def _load_vit(self, model_name: str, pretrained: bool):
        """Load Vision Transformer model."""
        try:
            import timm
            model = timm.create_model(model_name, pretrained=pretrained)
            return model
        except ImportError:
            print("⚠️  timm not installed. Install with: pip install timm")
            raise
    
    def _load_efficientnet(self, model_name: str, pretrained: bool):
        """Load EfficientNetV2 model."""
        try:
            import timm
            model = timm.create_model(model_name, pretrained=pretrained)
            return model
        except ImportError:
            print("⚠️  timm not installed. Install with: pip install timm")
            raise
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention-weighted ensemble.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Tuple of (logits, attention_weights)
        """
        # Extract features from both backbones
        vit_features = self.vit(x)  # (B, vit_features)
        eff_features = self.efficientnet(x)  # (B, eff_features)
        
        # Concatenate features
        combined = torch.cat([vit_features, eff_features], dim=1)  # (B, total_features)
        
        # Calculate attention weights
        attn_weights = self.attention(combined)  # (B, 2)
        
        # Apply attention to features
        weighted_vit = vit_features * attn_weights[:, 0:1]
        weighted_eff = eff_features * attn_weights[:, 1:2]
        weighted_combined = torch.cat([weighted_vit, weighted_eff], dim=1)
        
        # Fusion and classification
        fused = self.fusion(weighted_combined)
        logits = self.classifier(fused)
        
        return logits, attn_weights
    
    def predict(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Predict class probabilities and labels.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Dict with 'probabilities', 'labels', 'attention_weights'
        """
        self.eval()
        with torch.no_grad():
            logits, attn_weights = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            labels = torch.argmax(probabilities, dim=1)
        
        return {
            'probabilities': probabilities.cpu().numpy(),
            'labels': labels.cpu().numpy(),
            'attention_weights': attn_weights.cpu().numpy()
        }


class TextureFeatureExtractor(nn.Module):
    """
    Custom texture feature extractor for bite mark patterns.
    Uses Gabor filters and LBP (Local Binary Patterns) features.
    """
    
    def __init__(self, num_orientations: int = 8, num_scales: int = 4):
        """
        Args:
            num_orientations: Number of Gabor filter orientations
            num_scales: Number of Gabor filter scales
        """
        super().__init__()
        
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        
        # Create Gabor filter bank
        self.gabor_filters = self._create_gabor_bank()
        
    def _create_gabor_bank(self) -> List[np.ndarray]:
        """Create bank of Gabor filters."""
        filters = []
        
        for scale in range(self.num_scales):
            for orientation in range(self.num_orientations):
                theta = orientation * np.pi / self.num_orientations
                sigma = 2.0 ** scale
                lambd = sigma * 1.5
                gamma = 0.5
                
                kernel = cv2.getGaborKernel(
                    ksize=(31, 31),
                    sigma=sigma,
                    theta=theta,
                    lambd=lambd,
                    gamma=gamma,
                    psi=0,
                    ktype=cv2.CV_32F
                )
                
                filters.append(kernel)
        
        return filters
    
    def extract_gabor_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Gabor features from image.
        
        Args:
            image: Grayscale image (H, W)
            
        Returns:
            Gabor feature vector
        """
        features = []
        
        for kernel in self.gabor_filters:
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            features.append(filtered.mean())  # Mean response
            features.append(filtered.std())   # Std response
        
        return np.array(features)
    
    def extract_lbp_features(
        self, 
        image: np.ndarray, 
        radius: int = 3, 
        n_points: int = 24
    ) -> np.ndarray:
        """
        Extract Local Binary Pattern (LBP) histogram.
        
        Args:
            image: Grayscale image
            radius: LBP radius
            n_points: Number of circularly symmetric neighbor points
            
        Returns:
            LBP histogram feature vector
        """
        from skimage.feature import local_binary_pattern
        
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        
        # Compute histogram
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        # Normalize
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        return hist
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract texture features from batch of images.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Texture features (B, num_features)
        """
        batch_features = []
        
        for img in x:
            # Convert to grayscale numpy
            img_np = img.permute(1, 2, 0).cpu().numpy()
            gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Extract Gabor features
            gabor_feat = self.extract_gabor_features(gray)
            
            # Extract LBP features
            lbp_feat = self.extract_lbp_features(gray)
            
            # Concatenate
            combined = np.concatenate([gabor_feat, lbp_feat])
            batch_features.append(combined)
        
        return torch.tensor(batch_features, device=x.device, dtype=torch.float32)


class GradCAMVisualizer:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for model interpretability.
    Visualizes which regions the model focuses on for classification.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: Classification model
            target_layer: Layer to extract gradients from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self, 
        input_image: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_image: Input tensor (1, C, H, W)
            target_class: Target class index (None = predicted class)
            
        Returns:
            Heatmap (H, W) normalized to [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        logits, _ = self.model(input_image)
        
        # Get target class
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = logits[0, target_class]
        class_score.backward()
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ReLU (only positive influences)
        cam = F.relu(cam)
        
        # Resize to input size
        cam = F.interpolate(
            cam, 
            size=input_image.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize(
        self, 
        input_image: np.ndarray, 
        cam: np.ndarray, 
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            input_image: Original RGB image (H, W, 3)
            cam: Grad-CAM heatmap (H, W)
            alpha: Transparency of overlay
            
        Returns:
            Overlayed image
        """
        # Convert CAM to RGB heatmap
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        overlay = cv2.addWeighted(input_image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay


# Training configuration
class ClassificationConfig:
    """Configuration for training bite mark classifier."""
    
    # Model
    num_classes = 3
    vit_model = 'vit_base_patch16_224'
    efficientnet_model = 'efficientnetv2_m'
    
    # Training
    epochs = 200
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 1e-5
    
    # Optimizer
    optimizer = 'AdamW'
    scheduler = 'CosineAnnealingWarmRestarts'
    warmup_epochs = 10
    
    # Loss
    loss_fn = 'FocalLoss'  # Better for class imbalance
    focal_alpha = [1.0, 4.33, 0.62]  # Class weights from balanced training
    focal_gamma = 2.0
    
    # Regularization
    dropout = 0.3
    label_smoothing = 0.1
    mixup_alpha = 0.2
    cutmix_alpha = 1.0
    
    # Data augmentation
    augmentation = {
        'rotation': 180,
        'brightness': 0.3,
        'contrast': 0.3,
        'hue': 0.1,
        'saturation': 0.3,
        'noise': 0.05,
        'blur': 0.3
    }


# Example usage
if __name__ == '__main__':
    # Initialize classifier
    classifier = BiteMarkClassifier(
        num_classes=3,
        vit_model='vit_base_patch16_224',
        efficientnet_model='efficientnetv2_m',
        pretrained=True,
        dropout=0.3
    )
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = classifier.to(device)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    logits, attn_weights = classifier(dummy_input)
    
    print(f"✅ Classifier initialized successfully")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Attention weights: {attn_weights}")
    
    # Test Grad-CAM
    target_layer = classifier.vit.blocks[-1]  # Last transformer block
    grad_cam = GradCAMVisualizer(classifier, target_layer)
    
    cam = grad_cam.generate_cam(dummy_input)
    print(f"✅ Grad-CAM heatmap shape: {cam.shape}")
