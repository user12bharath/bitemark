# 🚀 Quick Reference - Balanced Training

## One-Line Commands

### Run Balanced Training (Recommended)
```bash
python scripts/run_balanced_train.py --regenerate-augmented
```

### Run with Balanced Batches (Experimental)
```bash
python scripts/run_balanced_train.py --regenerate-augmented --use-balanced-batches
```

### Run Unit Tests
```bash
python tests/test_balance.py
```

### Verify Class Balance
```bash
# Check raw dataset
python -c "from src.data_utils import verify_class_balance; verify_class_balance('data/raw', ['human', 'dog', 'snake'])"

# Check augmented dataset
python -c "from src.data_utils import verify_class_balance; verify_class_balance('data/augmented', ['human', 'dog', 'snake'])"
```

### View Results
```bash
# Metrics
cat outputs/metrics_balanced.json

# Confusion matrix (Windows)
start outputs/confusion_matrix_balanced.png

# Confusion matrix (macOS)
open outputs/confusion_matrix_balanced.png

# Confusion matrix (Linux)
xdg-open outputs/confusion_matrix_balanced.png
```

---

## File Locations

| File | Purpose |
|------|---------|
| `scripts/run_balanced_train.py` | Main training script |
| `src/data_utils.py` | Balanced batch generator, class weights |
| `src/augmentation.py` | Deterministic oversampling (SEED=42) |
| `tests/test_balance.py` | Unit tests |
| `models/balanced_model.h5` | Trained model (output) |
| `outputs/metrics_balanced.json` | Metrics (output) |
| `outputs/confusion_matrix_balanced.png` | Confusion matrix (output) |

---

## Expected Results

### Before Fix:
```
Raw dataset: human=24, dog=3, snake=35
Model predicts: human=0%, dog=0%, snake=100% ❌
```

### After Fix:
```
Augmented dataset: human=153, dog=153, snake=153
Model predicts: human≥60%, dog≥60%, snake≥60% ✅
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | `pip install scikit-learn matplotlib seaborn` |
| Data not regenerating | Delete `data/augmented/*` manually |
| Still low accuracy | Try `--use-balanced-batches` flag |
| Want more augmentation | Edit `augmentation_factor` in script |

---

## Key Parameters (in run_balanced_train.py)

```python
SEED = 42                      # Reproducibility
augmentation_factor = 3        # Augmentation multiplier
img_size = (224, 224)          # Image dimensions
batch_size = 16                # Batch size
epochs = 100                   # Max epochs (early stopping at ~45)
learning_rate = 0.001          # Initial learning rate
```

---

## Success Indicators

✅ Unit tests pass (4/4)  
✅ Augmented data balanced (equal counts)  
✅ Class weights logged correctly  
✅ Training completes in ~3-5 minutes  
✅ Model accuracy ≥70%  
✅ Per-class accuracy ≥60%  

---

## Next Steps After Training

1. Check `outputs/metrics_balanced.json` for detailed metrics
2. View `outputs/confusion_matrix_balanced.png` to verify no bias
3. Compare with baseline: `cat outputs/metrics.json` (old) vs `cat outputs/metrics_balanced.json` (new)
4. If dog accuracy still low, collect more real dog images
5. If results good, deploy `models/balanced_model.h5`

---

**Documentation:** See `README.md`, `TESTING_NOTES.md`, `DELIVERABLES.md`  
**Support:** Check `TESTING_NOTES.md` section "🐛 Known Limitations"
