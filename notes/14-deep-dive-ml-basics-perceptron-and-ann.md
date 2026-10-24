[Previous: 13. Quick Revision Cheat Sheet](13-quick-revision-cheat-sheet.md) | [Home](../README.md) | [Next: 15. Deep Dive: Computer Vision, OpenCV and CNN](15-deep-dive-computer-vision-opencv-and-cnn.md)

---

## 14. Deep Dive: ML Basics, Perceptron and ANN

### 14.1 Standard ML workflow

Collect data, clean, EDA, feature engineering, split, train, evaluate, tune, deploy. Typical split: **70% train, 15% validation, 15% test**. Test set ko sirf end mein ek baar use karo.

**Bias-variance tradeoff:** High bias = underfit (model bahut simple). High variance = overfit (train mein achha, test mein bura). Goal: dono ke beech sweet spot.

### 14.2 Evaluation metrics (formulas)

```math
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN},\quad \text{Precision}=\frac{TP}{TP+FP},\quad \text{Recall}=\frac{TP}{TP+FN}
```

```math
F1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
```

| Metric | Kab use karein |
| --- | --- |
| Accuracy | Balanced classes |
| Precision | False positive costly (spam filter) |
| Recall | False negative costly (disease detection, fraud) |
| F1 | Imbalanced data, precision-recall balance |
| ROC-AUC | Threshold-independent ranking quality |
| MAE / RMSE | Regression error (RMSE outliers ko zyada penalise karta hai) |
| R2 | Model kitna variance explain karta hai |

### 14.3 Perceptron worked example: AND gate

Weights w1 = 1, w2 = 1, bias b = -1.5, step activation (z > 0 gives 1).

| x1 | x2 | z = x1 + x2 - 1.5 | Output |
| --- | --- | --- | --- |
| 0 | 0 | -1.5 | 0 |
| 0 | 1 | -0.5 | 0 |
| 1 | 0 | -0.5 | 0 |
| 1 | 1 | 0.5 | 1 |

**XOR** ek straight line se alag nahi hota, isliye single perceptron fail hota hai. Hidden layer (MLP) XOR solve kar leta hai.

### 14.4 Forward pass: numeric example

Input x = [2, 3], weights w = [0.5, -1], bias b = 1.

`z = 0.5*2 + (-1)*3 + 1 = -1`

| Activation | Output for z = -1 |
| --- | --- |
| Sigmoid | 1/(1+e^1) = 0.269 |
| Tanh | -0.762 |
| ReLU | 0 |

### 14.5 Backpropagation (chain rule)

Backprop loss ka gradient har weight tak peeche ki taraf chain rule se pahunchata hai.

```math
\frac{\partial L}{\partial w}=\frac{\partial L}{\partial \hat y}\cdot\frac{\partial \hat y}{\partial z}\cdot\frac{\partial z}{\partial w}
```

Update: `w = w - lr * dL/dw`. Frameworks (TensorFlow, PyTorch) ye gradients automatically nikalte hain (autograd).

### 14.6 Activation functions in more detail

```math
\sigma(z)=\frac{1}{1+e^{-z}},\quad \sigma'(z)=\sigma(z)(1-\sigma(z))\le 0.25,\quad \text{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
```

- Sigmoid ka derivative max 0.25 hai. Kai layers multiply hone par gradient lagbhag 0 ho jata hai: **vanishing gradient**.
- **ReLU** ka derivative 1 (positive side) hai, isliye gradient flow achha. Negative side par 0, isse neuron 'mar' sakta hai (dying ReLU).
- Modern options: **ELU, GELU** (Transformers/BERT), **Swish**.
- Softmax outputs ka sum 1 hota hai, isliye probabilities jaisa behave karta hai.

### 14.7 Weight initialization

| Method | Kab |
| --- | --- |
| Xavier / Glorot | Sigmoid, tanh |
| He initialization | ReLU family (Keras mein `kernel_initializer='he_normal'`) |
| All zeros | **Kabhi nahi**: sab neurons same seekhte hain (symmetry problem) |

### 14.8 Loss functions: formulas and why

```math
MSE=\frac1n\sum(y-\hat y)^2,\quad BCE=-\frac1n\sum\big[y\log\hat y+(1-y)\log(1-\hat y)\big]
```

Classification mein **cross-entropy** isliye, kyunki wrong confident prediction ko bahut bada penalty milta hai aur gradient sigmoid/softmax ke saath achha behave karta hai. MSE classification par slow seekhta hai.

### 14.9 Optimizers in more detail

```math
\text{Momentum: } v_t=\beta v_{t-1}+\nabla L,\; w=w-\eta v_t
```

```math
\text{Adam: } m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\; v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2,\; w=w-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
```

Adam defaults: lr = 0.001, beta1 = 0.9, beta2 = 0.999.

| Batch size | Effect |
| --- | --- |
| Small (16-32) | Noisy updates, better generalisation, slower per epoch |
| Large (256+) | Stable, fast on GPU, may generalise worse |

**Learning rate schedules:** step decay, exponential decay, ReduceLROnPlateau, cosine annealing, warm-up.

### 14.10 Regularization in more detail

```math
L1:\; L+\lambda\sum|w|,\qquad L2:\; L+\lambda\sum w^2
```

- **Dropout** sirf training mein active, inference mein off. Typical rate 0.2-0.5.
- **BatchNorm** activations ko normalise karta hai: training fast, thoda regularising effect.
- **Early stopping** sabse sasta aur effective method.

### 14.11 Complete ANN example (tabular classification)

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow import keras

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit sirf train par
X_test = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(X_train.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy', keras.metrics.AUC(name='auc')])
hist = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32,
                 callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)], verbose=0)
print(classification_report(y_test, (model.predict(X_test) > 0.5).astype(int)))
```

### 14.12 Debugging checklist

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Loss NaN | Learning rate too high, bad scaling | Lower lr, scale inputs, clip gradients |
| Loss not decreasing | lr too low/high, bug, no scaling | Overfit 1 small batch first; check labels |
| Train good, val bad | Overfitting | Dropout, L2, more data, early stopping |
| Both bad | Underfitting | Bigger model, more epochs, better features |
| Val loss jumps around | lr high or small val set | Lower lr, bigger batch |

---

[Previous: 13. Quick Revision Cheat Sheet](13-quick-revision-cheat-sheet.md) | [Home](../README.md) | [Next: 15. Deep Dive: Computer Vision, OpenCV and CNN](15-deep-dive-computer-vision-opencv-and-cnn.md)
