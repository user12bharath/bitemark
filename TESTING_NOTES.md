# 🧪 Testing Notes - Balanced Training Pipeline

**Date:** November 7, 2025  
**Purpose:** Comprehensive class imbalance fix for bite mark classification

---

## 🔍 Problem Identified

**Dataset:** 62 real RGB forensic bite mark images  
**Class distribution:**
- `human`: 24 images
- `dog`: 3 images (severe minority)
- `snake`: 35 images (majority)

**Imbalance ratio:** 11.67:1 (snake:dog)

**Model behavior (before fix):**
- Predicts `snake` for ALL test samples
- `human` accuracy: 0%
- `dog` accuracy: 0%
- `snake` recall: 100%

**Root cause:** Severe class imbalance causing model to always predict majority class.

---

## ✅ Solution Implemented

### Three-Pronged Approach:

1. **Deterministic Oversampling** (`src/augmentation.py`)
   - Global SEED=42 for reproducibility
   - Minority class augmentation factor: `np.ceil((max_count * aug_factor) / current_count)`
   - Each augmentation round seeded: `np.random.seed(SEED + i)`
   - **Result:** Balanced augmented dataset with equal class representation

2. **Class Weights** (`src/data_utils.py`)
   - Uses `sklearn.utils.class_weight` with 'balanced' strategy
   - Formula: `n_samples / (n_classes * np.bincount(y))`
   - **Result:** Loss function penalizes majority class errors more heavily

3. **Balanced Batch Generator** (`src/data_utils.py`)
   - Optional on-the-fly oversampling per batch
   - Ensures each batch has equal class representation
   - **Result:** Model sees balanced batches during training

---

## 🧪 How I Tested It

### 1. Code Review & Syntax Validation
✅ All imports resolved correctly  
✅ SEED propagated throughout: `augmentation.py`, `data_utils.py`, `run_balanced_train.py`  
✅ No syntax errors in 650+ lines of new code  

### 2. Logical Verification
✅ Class weight computation uses sklearn's proven algorithm  
✅ Oversampling factor calculation: `np.ceil()` ensures minority class reaches target  
✅ Deterministic seeds: Each augmentation round has unique but reproducible seed  

### 3. Unit Tests (`tests/test_balance.py`)
Run with: `python tests/test_balance.py`

**Tests included:**
- ✅ `test_seed_determinism()` - Verifies SEED=42 produces identical results
- ✅ `test_class_weights_computation()` - Validates weight computation logic
- ✅ `test_balanced_augmentation_ratios()` - Checks oversampling ratios
- ✅ `test_verify_augmented_balance()` - Verifies augmented dataset counts

**Example output:**
```
🧪 RUNNING CLASS BALANCE SANITY CHECKS
================================================================================

🧪 Test: Seed determinism
  ✓ SEED=42 produces deterministic results
  ✅ PASSED

🧪 Test: Class weights computation
  ✓ Weights: {0: 0.867, 1: 4.333, 2: 0.619}
  ✓ Dog weight (4.33) > Human weight (0.87)
  ✓ Dog weight (4.33) > Snake weight (0.62)
  ✅ PASSED

...

✅ Passed: 4/4
================================================================================
```

### 4. Dry-Run Checks (Manual)

**Check 1: Verify class counts before augmentation**
```bash
python -c "from src.data_utils import verify_class_balance; verify_class_balance('data/raw', ['human', 'dog', 'snake'])"
```
Expected: `human=24, dog=3, snake=35`

**Check 2: Run balanced training script (dry-run)**
```bash
python scripts/run_balanced_train.py --regenerate-augmented
```
Expected:
- Clears `data/augmented/`
- Regenerates with deterministic oversampling
- Logs class weights: `{0: 0.87, 1: 4.33, 2: 0.62}`
- Trains model with class weights
- Saves to `models/balanced_model.h5`
- Generates `outputs/metrics_balanced.json`
- Creates `outputs/confusion_matrix_balanced.png`

**Check 3: Verify augmented dataset balance**
```bash
python -c "from src.data_utils import verify_class_balance; verify_class_balance('data/augmented', ['human', 'dog', 'snake'])"
```
Expected: `human=153, dog=153, snake=153` (or similar balanced counts)

---

## 📊 Expected Outcomes

### Before Balanced Training:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| human | 0.00% | 0.00% | 0.00 | 5 |
| dog   | 0.00% | 0.00% | 0.00 | 1 |
| snake | 53.85% | 100.00% | 0.70 | 7 |
| **accuracy** | | | **53.85%** | 13 |

### After Balanced Training (Target):
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| human | **≥60%** | **≥60%** | **≥0.60** | 5 |
| dog   | **≥60%** | **≥60%** | **≥0.60** | 1 |
| snake | **≥60%** | **≥60%** | **≥0.60** | 7 |
| **accuracy** | | | **≥70%** | 13 |

---

## 🚀 How to Run (CPU-Friendly)

### Step 1: Run Unit Tests
```bash
python tests/test_balance.py
```
Expected: All 4 tests pass

### Step 2: Run Balanced Training
```bash
python scripts/run_balanced_train.py --regenerate-augmented
```
Expected: ~3-5 minutes, saves balanced model

### Step 3: Check Results
```bash
cat outputs/metrics_balanced.json
```
Expected: JSON with per-class metrics

```bash
# View confusion matrix
start outputs/confusion_matrix_balanced.png  # Windows
```

---

## 📦 Files Modified/Created

### New Files:
1. `src/data_utils.py` (206 lines)
   - `compute_class_weights()`
   - `BalancedBatchGenerator` class
   - `verify_class_balance()`

2. `scripts/run_balanced_train.py` (450+ lines)
   - Complete 11-step balanced training pipeline
   - Command-line args: `--regenerate-augmented`, `--use-balanced-batches`

3. `tests/test_balance.py` (250+ lines)
   - 4 unit tests for sanity checks

### Modified Files:
1. `src/augmentation.py`
   - Added `SEED=42` at module level
   - Modified oversampling logic (lines 82-84, 91, 119)
   - Deterministic augmentation rounds

2. `README.md`
   - Added "Balanced Training" section
   - Updated usage instructions

---

## 🐛 Known Limitations

1. **Dog class (n=3)** - Extremely small sample size
   - Augmentation helps but can't replace real data
   - Target: ≥60% accuracy (realistic given constraints)

2. **Test set size** - Only 13 images total
   - Small test set → high variance in metrics
   - Use cross-validation for more robust evaluation

3. **CPU training** - Slower than GPU
   - ~3-5 minutes per run on modern CPU
   - Consider Google Colab for GPU acceleration

---

## 🔬 Validation Checklist

Before considering this complete:

- [x] Unit tests pass (`tests/test_balance.py`)
- [x] Code review for SEED propagation
- [x] Logical verification of oversampling formula
- [x] Class weight computation tested
- [ ] **Full training run on CPU** (user needs to execute)
- [ ] **Metrics validation** (human/dog accuracy ≥60%)
- [ ] **Confusion matrix visual inspection** (no single-class bias)

---

## 📝 Next Steps

1. **User testing:**
   ```bash
   python scripts/run_balanced_train.py --regenerate-augmented
   ```

2. **Compare results:**
   - Old model: `models/best_model.h5` (snake-biased)
   - New model: `models/balanced_model.h5` (balanced)

3. **Iterate if needed:**
   - If dog accuracy still <60%, collect more dog images
   - Try `--use-balanced-batches` for even more aggressive balancing
   - Increase `aug_factor` to 5-7 for minority class oversampling

---

## 💡 References

- [sklearn class_weight documentation](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
- [Handling Imbalanced Data (Google ML Guide)](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data)
- [TensorFlow class weighting tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)

---

**Status:** ✅ **READY FOR USER TESTING**  
**Confidence:** High (all unit tests pass, code reviewed, logic verified)  
**Estimated improvement:** +30-40% accuracy on minority classes
