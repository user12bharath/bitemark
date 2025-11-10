"""
BITE MARK CLASSIFICATION - DEMO RUNNER
A simplified version that can run with or without GPU
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🦷 BITE MARK CLASSIFICATION - DEMO MODE")
print("="*80)

# Check for dependencies
try:
    import numpy as np
    print("✓ NumPy available")
except ImportError:
    print("❌ NumPy not installed. Run: pip install numpy")
    sys.exit(1)

try:
    import tensorflow as tf
    print(f"✓ TensorFlow available (v{tf.__version__})")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠ No GPU detected - using CPU (slower but works)")
        
except ImportError:
    print("❌ TensorFlow not installed. Run: pip install tensorflow")
    print("   For CPU-only: pip install tensorflow-cpu")
    sys.exit(1)

try:
    import cv2
    print("✓ OpenCV available")
except ImportError:
    print("⚠ OpenCV not available - using basic image loading")
    cv2 = None

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib available")
except ImportError:
    print("❌ Matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    print("✓ Scikit-learn available")
except ImportError:
    print("❌ Scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

print("\n" + "="*80)
print("All dependencies satisfied! Running pipeline...")
print("="*80 + "\n")

# Run the main pipeline
exec(open('main_pipeline.py').read())
