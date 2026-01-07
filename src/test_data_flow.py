"""
Test script to verify data flow and file connections
Checks if all components are properly loading augmented data
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from data_preprocessing import BiteMarkPreprocessor, PreprocessingConfig
from augmentation import BiteMarkAugmentor, AugmentationConfig


def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_augmented_data_exists():
    """Test if augmented data exists"""
    print_section("TEST 1: Check Augmented Data Existence")
    
    augmented_dir = '../data/augmented'
    
    if not os.path.exists(augmented_dir):
        print(f"❌ FAIL: Augmented directory not found: {augmented_dir}")
        return False
    
    print(f"✓ Augmented directory exists: {augmented_dir}")
    
    # Count files in each class
    classes = ['dog', 'human', 'snake']
    total_files = 0
    
    for class_name in classes:
        class_path = os.path.join(augmented_dir, class_name)
        if os.path.exists(class_path):
            files = [f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            count = len(files)
            total_files += count
            print(f"  {class_name}: {count} images")
        else:
            print(f"  ❌ {class_name}: directory not found")
    
    print(f"\n✓ Total augmented images: {total_files}")
    return total_files > 0


def test_preprocessor_loading():
    """Test if preprocessor loads augmented data correctly"""
    print_section("TEST 2: Preprocessor Data Loading")
    
    try:
        config = PreprocessingConfig(
            img_size=(224, 224),
            grayscale=True,
            normalize=True
        )
        preprocessor = BiteMarkPreprocessor(config=config)
        
        # Test with explicit augmented path
        images, labels, class_names = preprocessor.load_sample_data(
            data_dir='../data/augmented'
        )
        
        print(f"✓ Data loaded successfully")
        print(f"  Total images: {len(images)}")
        print(f"  Image shape: {images[0].shape}")
        print(f"  Classes: {class_names}")
        
        # Check class distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n  Class distribution:")
        for cls_idx, count in zip(unique, counts):
            print(f"    {class_names[cls_idx]}: {count} images")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessor_default():
    """Test if preprocessor uses augmented data by default"""
    print_section("TEST 3: Preprocessor Default Behavior")
    
    try:
        preprocessor = BiteMarkPreprocessor(img_size=(224, 224), grayscale=True)
        
        # Load without specifying directory (should use default)
        images, labels, class_names = preprocessor.load_sample_data()
        
        print(f"✓ Default loading successful")
        print(f"  Loaded {len(images)} images")
        print(f"  Classes: {class_names}")
        
        # Verify it's loading augmented data (should have >100 images)
        if len(images) > 100:
            print(f"✓ PASS: Loaded augmented data by default (>100 images)")
            return True
        else:
            print(f"⚠ WARNING: Only {len(images)} images loaded (expected >100 for augmented)")
            print(f"  This might be loading raw data instead")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_augmentation_consistency():
    """Test if augmentation produces consistent results"""
    print_section("TEST 4: Augmentation Consistency")
    
    try:
        # Create a simple test image
        test_image = np.random.rand(224, 224, 1).astype(np.float32)
        
        config = AugmentationConfig()
        augmentor = BiteMarkAugmentor(preserve_features=True, config=config)
        
        # Apply augmentation twice with same seed
        np.random.seed(42)
        aug1 = augmentor.apply_random_augmentation(test_image)
        
        np.random.seed(42)
        aug2 = augmentor.apply_random_augmentation(test_image)
        
        # Check if results are identical
        if np.allclose(aug1, aug2):
            print("✓ PASS: Augmentation is deterministic")
            return True
        else:
            print("❌ FAIL: Augmentation produces different results with same seed")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_train_pipeline_compatibility():
    """Test if train pipeline can use the data"""
    print_section("TEST 5: Training Pipeline Compatibility")
    
    try:
        from data_preprocessing import BiteMarkPreprocessor, PreprocessingConfig
        
        config = PreprocessingConfig(
            img_size=(224, 224),
            grayscale=True,
            normalize=True,
            adaptive_histogram=True,
            denoise=True
        )
        
        preprocessor = BiteMarkPreprocessor(config=config)
        images, labels, class_names = preprocessor.load_sample_data(
            data_dir='../data/augmented'
        )
        
        # Test data splitting
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            images, labels, test_size=0.2, val_size=0.1
        )
        
        print(f"✓ Data splitting successful")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Test TF dataset creation
        import tensorflow as tf
        train_dataset = preprocessor.create_tf_dataset(
            X_train, y_train, batch_size=16, shuffle=True, augment=False
        )
        
        # Try to get one batch
        for batch_x, batch_y in train_dataset.take(1):
            print(f"✓ TF Dataset created successfully")
            print(f"  Batch shape: {batch_x.shape}")
            print(f"  Label shape: {batch_y.shape}")
            break
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  BITEMARK DATA FLOW TEST SUITE")
    print("="*60)
    
    tests = [
        ("Augmented Data Exists", test_augmented_data_exists),
        ("Preprocessor Loading", test_preprocessor_loading),
        ("Preprocessor Default", test_preprocessor_default),
        ("Augmentation Consistency", test_data_augmentation_consistency),
        ("Training Pipeline", test_train_pipeline_compatibility),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Data flow is properly configured!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed - Review the issues above")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
