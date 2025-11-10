# 🦷 Bite Mark Classification - Training Summary Report

**Generated on:** 2025-11-10 13:39:13

---

## 📊 Dataset Statistics

- **Total Samples:** 62
- **Training Samples:** 207
- **Validation Samples:** 10
- **Test Samples:** 13
- **Classes:** ['human', 'dog', 'snake']
- **Image Size:** (224, 224)

---

## ⚙️ Training Configuration

- **Training Duration:** 465.77 seconds (7.76 minutes)
- **Epochs Completed:** 84
- **Batch Size:** 8
- **Optimizer:** Adam
- **Learning Rate:** 0.0005
- **GPU Acceleration:** No
- **Mixed Precision:** Enabled (FP16)

---

## 🎯 Model Performance

### Final Metrics
- **Test Accuracy:** 76.92%
- **Test Loss:** 0.6546

### Per-Class Metrics
```
              precision    recall  f1-score   support

       human      0.625     1.000     0.769         5
         dog      0.000     0.000     0.000         1
       snake      1.000     0.714     0.833         7

    accuracy                          0.769        13
   macro avg      0.542     0.571     0.534        13
weighted avg      0.779     0.769     0.745        13

```

### Training Progress
- **Best Validation Accuracy:** 100.00%
- **Final Training Accuracy:** 97.10%
- **Final Validation Accuracy:** 100.00%

---

## 📈 Confusion Matrix Summary

```
[[5 0 0]
 [1 0 0]
 [2 0 5]]
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

- **GPU Model:** CPU
- **Memory Optimization:** Mixed Precision (FP16)
- **Batch Size Optimization:** Adaptive based on 4GB VRAM

---

## ✅ Conclusion

The bite mark classification model has been successfully trained and evaluated.
Review the visualizations and metrics for detailed performance analysis.

