"""Quick image analysis script"""
import cv2
import numpy as np
import os
from pathlib import Path

print("="*80)
print("ANALYZING EXISTING BITE MARK IMAGES")
print("="*80)

# Count images per class
classes = ['human', 'cat', 'dog', 'snake']
class_counts = {}

for cls in classes:
    path = f'data/raw/{cls}'
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        class_counts[cls] = len(files)
        print(f"\n{cls.upper()}: {len(files)} images")
    else:
        class_counts[cls] = 0
        print(f"\n{cls.upper()}: 0 images")

total = sum(class_counts.values())
print(f"\nTOTAL IMAGES: {total}")

# Analyze sample images from each class
print("\n" + "="*80)
print("SAMPLE IMAGE ANALYSIS")
print("="*80)

for cls in classes:
    path = f'data/raw/{cls}'
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if files:
            sample_path = os.path.join(path, files[0])
            img = cv2.imread(sample_path)
            if img is not None:
                print(f"\n{cls.upper()} - {files[0]}:")
                print(f"  Shape: {img.shape}")
                print(f"  Size: {img.shape[1]}x{img.shape[0]} pixels")
                print(f"  Channels: {img.shape[2] if len(img.shape) > 2 else 1}")
                print(f"  Dtype: {img.dtype}")
                print(f"  Range: [{img.min()}, {img.max()}]")
                print(f"  Mean: {img.mean():.2f}")
                
                # Check if color or grayscale
                if len(img.shape) == 3 and img.shape[2] == 3:
                    b, g, r = cv2.split(img)
                    if np.array_equal(b, g) and np.array_equal(g, r):
                        print(f"  Type: Grayscale (stored as RGB)")
                    else:
                        print(f"  Type: Color (RGB)")

# Class imbalance analysis
print("\n" + "="*80)
print("CLASS IMBALANCE ANALYSIS")
print("="*80)

if total > 0:
    for cls in classes:
        percentage = (class_counts[cls] / total) * 100
        print(f"{cls.upper():>8}: {class_counts[cls]:3} images ({percentage:5.1f}%)")
    
    # Check imbalance ratio
    max_count = max(class_counts.values())
    min_count = min([v for v in class_counts.values() if v > 0]) if any(v > 0 for v in class_counts.values()) else 1
    imbalance_ratio = max_count / min_count if min_count > 0 else 0
    
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 3:
        print("⚠️  SEVERE CLASS IMBALANCE DETECTED!")
        print("   Recommendations:")
        print("   1. Use class weights during training")
        print("   2. Apply more augmentation to minority classes")
        print("   3. Consider SMOTE or other balancing techniques")
    elif imbalance_ratio > 1.5:
        print("⚠️  Moderate class imbalance detected")
        print("   Recommendation: Use class weights")
    else:
        print("✓ Classes are relatively balanced")

# Missing class handling
print("\n" + "="*80)
print("MISSING CLASS HANDLING")
print("="*80)

missing_classes = [cls for cls in classes if class_counts[cls] == 0]
if missing_classes:
    print(f"⚠️  Missing classes: {', '.join(missing_classes)}")
    print("   Options:")
    print("   1. Remove missing classes from training")
    print("   2. Collect more data for missing classes")
    print("   3. Use synthetic data generation for missing classes")
else:
    print("✓ All classes have data")

print("\n" + "="*80)
