[Previous: 12. Stock Price Prediction using Sequence Models (LSTM)](12-stock-price-prediction-using-sequence-models-lst.md) | [Home](../README.md) | [Next: 14. Deep Dive: ML Basics, Perceptron and ANN](14-deep-dive-ml-basics-perceptron-and-ann.md)

---

## 13. Quick Revision Cheat Sheet

| Problem | Go-to approach |
| --- | --- |
| Tabular classification/regression | ML models (XGBoost, RF) ya MLP |
| Image classification | CNN / transfer learning (ResNet, MobileNet) |
| Object detection | YOLO / Faster R-CNN |
| Pixel-level labelling | U-Net / DeepLab |
| Text classification | TF-IDF + LR, ya LSTM / BERT |
| Sequence / text generation | LSTM / GRU / Transformer |
| Seasonal time series | SARIMA / Prophet / LSTM |
| Extract text from images | OCR (Tesseract) |

**Top interview questions (short answers)**

1. **Overfitting kya hai, kaise roken?** Model train data memorise kar le; Dropout, L1/L2, early stopping, more data, augmentation.
2. **Why non-linear activation?** Bina uske deep network bhi linear model hi hai.
3. **ReLU kyun popular?** Simple, fast, vanishing gradient kam; dying ReLU ka risk.
4. **Why LSTM over RNN?** Gates + cell state se long-term dependency handle hoti hai.
5. **CNN vs ANN for images?** Weight sharing, local patterns, kam parameters.
6. **Stationarity kyun zaruri?** ARIMA jaise models stable statistical properties assume karte hain.
7. **Stemming vs Lemmatization?** Stemming crude cut; lemmatization dictionary-based valid root.
8. **TF-IDF kya hai?** Word importance = frequency x rarity across documents.
9. **Accuracy ke alawa kya?** Imbalanced data mein Precision, Recall, F1, ROC-AUC.
10. **Time series mein random split kyun nahi?** Future data leak ho jata hai; hamesha time-based split.

**Study order suggestion:** 1 -> 2 -> 3 (foundation, sabse important) -> 6 -> 7 -> 8 -> 10 -> 11. Data Analyst / Data Scientist interviews mein sections 3, 7, 10, 11 aur evaluation metrics sabse zyada poochhe jate hain.

---

# Part 2: Deep Dive Notes (details, math, worked examples)

Part 1 mein concepts aur diagrams the. Yahan har module ko gehrai mein samjhaya hai: formulas, numeric examples, extra code, common mistakes aur interview points.

---

[Previous: 12. Stock Price Prediction using Sequence Models (LSTM)](12-stock-price-prediction-using-sequence-models-lst.md) | [Home](../README.md) | [Next: 14. Deep Dive: ML Basics, Perceptron and ANN](14-deep-dive-ml-basics-perceptron-and-ann.md)
