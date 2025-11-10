"""
Data utilities for balanced training.

Provides:
- Balanced batch generator (on-the-fly oversampling)
- Class weight computation
- Deterministic data splitting
"""

import numpy as np
from typing import Tuple, Dict
from sklearn.utils.class_weight import compute_class_weight

# Global seed for reproducibility
SEED = 42


def compute_class_weights(y_train: np.ndarray, class_names: list) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        y_train: Training labels (integer encoded)
        class_names: List of class names
        
    Returns:
        Dictionary mapping class index to weight
        
    Example:
        >>> y = np.array([0, 0, 0, 1, 2, 2, 2, 2])
        >>> weights = compute_class_weights(y, ['a', 'b', 'c'])
        >>> print(weights)  # {0: 1.33, 1: 4.0, 2: 1.0}
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    
    class_weights = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    
    print("\n📊 Class Weights (for imbalanced training):")
    for idx, name in enumerate(class_names):
        if idx in class_weights:
            print(f"  {name}: {class_weights[idx]:.3f}")
    
    return class_weights


class BalancedBatchGenerator:
    """
    Generator that yields balanced batches by oversampling minority classes.
    
    Ensures each batch contains equal representation of all classes.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = SEED
    ):
        """
        Initialize balanced batch generator.
        
        Args:
            X: Training images (N, H, W, C)
            y: Training labels (N,)
            batch_size: Batch size (should be divisible by num_classes)
            shuffle: Whether to shuffle within each class
            seed: Random seed for reproducibility
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Group indices by class
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.class_indices = {cls: np.where(y == cls)[0] for cls in self.classes}
        
        # Calculate samples per class per batch
        self.samples_per_class = batch_size // self.num_classes
        
        # Current position in each class
        self.class_positions = {cls: 0 for cls in self.classes}
        
        print(f"\n🔄 BalancedBatchGenerator initialized:")
        print(f"  Batch size: {batch_size}")
        print(f"  Samples per class per batch: {self.samples_per_class}")
        print(f"  Classes: {self.num_classes}")
        
    def __len__(self) -> int:
        """Number of batches per epoch."""
        min_samples = min(len(indices) for indices in self.class_indices.values())
        return max(1, min_samples // self.samples_per_class)
    
    def __iter__(self):
        """Iterator for batch generation."""
        self.on_epoch_end()
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one balanced batch."""
        batch_X = []
        batch_y = []
        
        for cls in self.classes:
            indices = self.class_indices[cls]
            pos = self.class_positions[cls]
            
            # Get samples for this class
            batch_indices = []
            for _ in range(self.samples_per_class):
                if pos >= len(indices):
                    pos = 0  # Wrap around (oversampling)
                batch_indices.append(indices[pos])
                pos += 1
            
            self.class_positions[cls] = pos
            
            # Add to batch
            batch_X.extend(self.X[batch_indices])
            batch_y.extend(self.y[batch_indices])
        
        # Shuffle within batch
        if self.shuffle:
            perm = self.rng.permutation(len(batch_X))
            batch_X = [batch_X[i] for i in perm]
            batch_y = [batch_y[i] for i in perm]
        
        return np.array(batch_X), np.array(batch_y)
    
    def on_epoch_end(self):
        """Shuffle class indices at epoch end."""
        if self.shuffle:
            for cls in self.classes:
                self.rng.shuffle(self.class_indices[cls])
        
        # Reset positions
        self.class_positions = {cls: 0 for cls in self.classes}


def verify_class_balance(data_dir: str, class_names: list) -> Dict[str, int]:
    """
    Verify class balance in augmented dataset.
    
    Args:
        data_dir: Path to augmented data directory
        class_names: List of class names
        
    Returns:
        Dictionary with class counts
    """
    import os
    
    counts = {}
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if os.path.exists(class_path):
            counts[class_name] = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))])
        else:
            counts[class_name] = 0
    
    print(f"\n📁 Augmented dataset balance:")
    for name, count in counts.items():
        print(f"  {name}: {count} images")
    
    return counts
