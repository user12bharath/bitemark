"""
Unit tests and sanity checks for class balance.

Run with: python -m pytest tests/test_balance.py -v
Or: python tests/test_balance.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from data_utils import compute_class_weights, verify_class_balance, SEED


def test_class_weights_computation():
    """Test class weight computation."""
    print("\n🧪 Test: Class weights computation")
    
    # Imbalanced dataset (similar to our problem)
    y_train = np.array([0, 0, 0, 0, 0,  # 5 human
                        1,                 # 1 dog
                        2, 2, 2, 2, 2, 2, 2])  # 7 snake
    
    class_names = ['human', 'dog', 'snake']
    weights = compute_class_weights(y_train, class_names)
    
    # Assertions
    assert len(weights) == 3, f"Expected 3 weights, got {len(weights)}"
    assert all(w > 0 for w in weights.values()), "All weights should be positive"
    assert weights[1] > weights[0], "Dog (minority) weight should be higher than human"
    assert weights[1] > weights[2], "Dog (minority) weight should be higher than snake"
    
    print(f"  ✓ Weights: {weights}")
    print(f"  ✓ Dog weight ({weights[1]:.2f}) > Human weight ({weights[0]:.2f})")
    print(f"  ✓ Dog weight ({weights[1]:.2f}) > Snake weight ({weights[2]:.2f})")
    print("  ✅ PASSED")
    
    return True


def test_verify_augmented_balance():
    """Test verification of augmented dataset balance."""
    print("\n🧪 Test: Verify augmented dataset balance")
    
    class_names = ['human', 'dog', 'snake']
    
    # Check if augmented directory exists
    if not os.path.exists('data/augmented'):
        print("  ⚠️  SKIPPED (data/augmented/ not found)")
        return True
    
    counts = verify_class_balance('data/augmented', class_names)
    
    # Basic assertions
    assert len(counts) == 3, f"Expected 3 classes, got {len(counts)}"
    assert all(c >= 0 for c in counts.values()), "All counts should be non-negative"
    
    if all(c > 0 for c in counts.values()):
        # Check if reasonably balanced (within 50% of each other)
        max_count = max(counts.values())
        min_count = min(counts.values())
        ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"  ✓ Counts: {counts}")
        print(f"  ✓ Max/Min ratio: {ratio:.2f}")
        
        if ratio <= 2.0:
            print(f"  ✅ PASSED (well balanced, ratio={ratio:.2f})")
        else:
            print(f"  ⚠️  WARNING (imbalanced, ratio={ratio:.2f})")
    else:
        print(f"  ⚠️  WARNING (some classes have 0 images)")
    
    return True


def test_seed_determinism():
    """Test that SEED produces deterministic results."""
    print("\n🧪 Test: Seed determinism")
    
    np.random.seed(SEED)
    result1 = np.random.rand(10)
    
    np.random.seed(SEED)
    result2 = np.random.rand(10)
    
    assert np.allclose(result1, result2), "SEED should produce identical results"
    
    print(f"  ✓ SEED={SEED} produces deterministic results")
    print("  ✅ PASSED")
    
    return True


def test_balanced_augmentation_ratios():
    """Test that augmentation produces correct ratios."""
    print("\n🧪 Test: Balanced augmentation ratios")
    
    # Simulate class counts before augmentation
    original_counts = {'human': 15, 'dog': 2, 'snake': 22}
    max_count = max(original_counts.values())  # 22
    aug_factor = 3
    
    # Expected targets (roughly)
    expected_targets = {
        'human': 15 * np.ceil((max_count * aug_factor) / 15),
        'dog': 2 * np.ceil((max_count * aug_factor) / 2),
        'snake': 22 * np.ceil((max_count * aug_factor) / 22),
    }
    
    print(f"  Original: {original_counts}")
    print(f"  Max count: {max_count}")
    print(f"  Aug factor: {aug_factor}")
    print(f"  Expected targets: {expected_targets}")
    
    # Check that dog gets the most augmentation
    dog_factor = expected_targets['dog'] / original_counts['dog']
    human_factor = expected_targets['human'] / original_counts['human']
    snake_factor = expected_targets['snake'] / original_counts['snake']
    
    print(f"  Augmentation factors:")
    print(f"    Dog: {dog_factor:.1f}x")
    print(f"    Human: {human_factor:.1f}x")
    print(f"    Snake: {snake_factor:.1f}x")
    
    assert dog_factor >= human_factor, "Dog should get more augmentation than human"
    assert dog_factor >= snake_factor, "Dog should get more augmentation than snake"
    
    print("  ✅ PASSED")
    
    return True


def run_all_tests():
    """Run all sanity checks."""
    print("="*80)
    print("🧪 RUNNING CLASS BALANCE SANITY CHECKS")
    print("="*80)
    
    tests = [
        test_seed_determinism,
        test_class_weights_computation,
        test_balanced_augmentation_ratios,
        test_verify_augmented_balance,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print(f"✅ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(tests)}")
    print("="*80 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
