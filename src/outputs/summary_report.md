# 🦷 Bite Mark Classification - Training Summary Report

**Generated on:** 2025-12-15 23:54:57

---

## 📊 Dataset Statistics

- **Total Samples:** 348
- **Training Samples:** 641
- **Validation Samples:** 35
- **Test Samples:** 70
- **Classes:** ['dog', 'human', 'snake']
- **Image Size:** (224, 224)

---

## ⚙️ Training Configuration

- **Training Duration:** 205.96 seconds (3.43 minutes)
- **Epochs Completed:** 1
- **Batch Size:** 16
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **GPU Acceleration:** Yes
- **Mixed Precision:** Enabled (FP16)

---

## 🎯 Model Performance

### Final Metrics
- **Test Accuracy:** 62.86%
- **Test Loss:** 1.2589

### Per-Class Metrics
```
              precision    recall  f1-score   support

       human      0.621     0.857     0.720        21
         dog      0.634     1.000     0.776        26
       snake      0.000     0.000     0.000        23

    accuracy                          0.629        70
   macro avg      0.418     0.619     0.499        70
weighted avg      0.422     0.629     0.504        70

```

### Training Progress
- **Best Validation Accuracy:** 62.86%
- **Final Training Accuracy:** 62.86%
- **Final Validation Accuracy:** 62.86%

---

## 📈 Confusion Matrix Summary

```
[[18  3  0]
 [ 0 26  0]
 [11 12  0]]
```

---

## 💾 Model Artifacts

- **Best Model:** `models/best_model.h5`
- **Training History Plot:** `outputs/training_history.png`
- **Confusion Matrix:** `outputs/confusion_matrix.png`
- **Sample Predictions:** `outputs/sample_predictions.png`
- **Metrics JSON:** `outputs/metrics.json`

---

## 🚀 Hardware Utilization

- **GPU Model:** 4GB RTX (Optimized)
- **Memory Optimization:** Mixed Precision (FP16)
- **Batch Size Optimization:** Adaptive based on 4GB VRAM

---

## ✅ Conclusion

The bite mark classification model has been successfully trained and evaluated.
Review the visualizations and metrics for detailed performance analysis.

