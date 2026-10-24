[Previous: 7. NLP: Sentiment Analysis on Amazon Reviews](07-nlp-sentiment-analysis-on-amazon-reviews.md) | [Home](../README.md) | [Next: 9. Chatbot](09-chatbot.md)

---

## 8. Sequence Models and Fake News Detection

### 8.1 RNN (Recurrent Neural Network)

Sequence data (text, time series, speech) ke liye. **Hidden state** pichle steps ki memory rakhta hai.

<p align="center"><img src="../images/11-rnn-unrolled.svg" alt="RNN unrolled over 3 time steps" width="800"><br><sub><i>RNN unrolled over 3 time steps</i></sub></p>

**Types:** one-to-one, one-to-many (image captioning), many-to-one (sentiment), many-to-many (translation).

**Problem:** Vanishing gradient, isliye long-term dependencies yaad nahi rehti. **Training:** BPTT (Backpropagation Through Time).

### 8.2 LSTM (Long Short-Term Memory)

Cell state (conveyor belt) + 3 gates se long memory.

<p align="center"><img src="../images/12-lstm-cell.svg" alt="LSTM cell · 3 gates and a cell-state line" width="800"><br><sub><i>LSTM cell · 3 gates and a cell-state line</i></sub></p>

| Gate | Kaam |
| --- | --- |
| Forget gate | Purani info kitni bhoolni hai |
| Input gate | Nayi info kitni store karni hai |
| Output gate | Cell se kitna output dena hai |

### 8.3 GRU (Gated Recurrent Unit)

LSTM ka simplified version: **2 gates** (Update, Reset), cell state nahi.

|  | RNN | LSTM | GRU |
| --- | --- | --- | --- |
| Gates | 0 | 3 | 2 |
| Long memory | Weak | Strong | Strong |
| Speed / parameters | Fastest | Slowest, most | Faster than LSTM |

**Bidirectional RNN:** sequence ko aage aur peeche dono taraf se padhta hai (context behtar).

### 8.4 Fake News Detection

Binary classification: Real vs Fake.

<p align="center"><img src="../images/13-fake-news-pipeline.svg" alt="fake news detection pipeline" width="800"><br><sub><i>fake news detection pipeline</i></sub></p>

```python
model = keras.Sequential([
    keras.layers.Embedding(10000, 64, input_length=200),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

**Baseline compare karo:** TF-IDF + Logistic Regression / Passive Aggressive Classifier. **Metrics:** Precision, Recall, F1, Confusion Matrix. **Caution:** data leakage (source name / dataset artifacts) se fake high accuracy aa sakti hai.

---

[Previous: 7. NLP: Sentiment Analysis on Amazon Reviews](07-nlp-sentiment-analysis-on-amazon-reviews.md) | [Home](../README.md) | [Next: 9. Chatbot](09-chatbot.md)
