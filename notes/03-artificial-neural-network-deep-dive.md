[Previous: 2. Introduction to Deep Learning](02-introduction-to-deep-learning.md) | [Home](../README.md) | [Next: 4. Computer Vision](04-computer-vision.md)

---

## 3. Artificial Neural Network Deep Dive

### 3.1 Training flow (yaad rakhne wala)

<p align="center"><img src="../images/05-training-loop.png" alt="training loop · 6 steps" width="800"><br><sub><i>training loop · 6 steps</i></sub></p>

Ek **epoch** = poora dataset ek baar. **Batch** = ek baar mein kitna data. **Iteration** = ek batch update.

### 3.2 Activation Functions

Non-linearity add karte hain, warna network sirf linear regression ban jata.

| Function | Formula | Range | Use | Problem |
| --- | --- | --- | --- | --- |
| Sigmoid | 1/(1+e^-x) | 0 to 1 | Binary output layer | Vanishing gradient |
| Tanh | (e^x - e^-x)/(e^x + e^-x) | -1 to 1 | Hidden (RNN) | Vanishing gradient |
| ReLU | max(0, x) | 0 to inf | Default for hidden layers | Dying ReLU |
| Leaky ReLU | max(0.01x, x) | -inf to inf | Fixes dying ReLU | Slope to tune |
| Softmax | e^xi / sum(e^xj) | 0 to 1, sum = 1 | Multi-class output | - |

**Rule of thumb:** Hidden = ReLU, Binary output = Sigmoid, Multi-class output = Softmax, Regression output = Linear.

### 3.3 Loss Functions

| Task | Loss | Formula idea |
| --- | --- | --- |
| Regression | MSE | mean of (y - y\_hat)^2 (outliers ko zyada penalise) |
| Regression | MAE | mean of abs(y - y\_hat) (outlier-robust) |
| Regression | Huber | MSE + MAE ka mix |
| Binary classification | Binary Cross-Entropy | -[y log(p) + (1-y) log(1-p)] |
| Multi-class (one-hot) | Categorical Cross-Entropy | -sum(y log p) |
| Multi-class (integer labels) | Sparse Categorical CE | Same, but integer labels |

### 3.4 Optimization Techniques

Goal: loss minimum karna. **Gradient Descent:** `w = w - lr * dLoss/dw`

```
Loss
 |\
 | \        <- learning rate bada = overshoot
 |  \__
 |     \_/   <- minimum
 +----------- weights
```

| Optimizer | Idea |
| --- | --- |
| Batch GD | Poora data ek step ke liye (slow, stable) |
| SGD | Ek sample (fast, noisy) |
| Mini-batch GD | Chhote batches (practical default) |
| Momentum | Past gradients ka average, jaldi converge |
| RMSprop | Per-parameter adaptive learning rate |
| Adagrad | Rare features ko bada update |
| Adam | Momentum + RMSprop (sabse popular default) |

**Learning rate:** bahut chhota = slow, bahut bada = diverge.

### 3.5 Regularization Techniques

Overfitting rokne ke liye (train accuracy high, test accuracy low).

```
Underfit          Good fit          Overfit
  ----             ~~~~             /\/\/\
(too simple)    (just right)      (memorised noise)
```

| Technique | How it works |
| --- | --- |
| L1 (Lasso) | Penalty = sum of abs(w); kuch weights zero ho jate hain |
| L2 (Ridge / weight decay) | Penalty = sum of w^2; weights chhote rakhta hai |
| Dropout | Training mein random neurons off (e.g. 0.5) |
| Early Stopping | Val loss badhne lage to training rok do |
| Data Augmentation | Flip/rotate/zoom se zyada data |
| Batch Normalization | Layer inputs normalise, training stable |

### 3.6 Callbacks (Keras)

Training ke dauran automatically chalne wale functions.

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best.keras', monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
]
model.fit(X, y, epochs=100, validation_split=0.2, callbacks=callbacks)
```

Other: TensorBoard (visualisation), LearningRateScheduler, CSVLogger, custom callbacks.

### 3.7 Problems to remember

- **Vanishing gradient:** gradients bahut chhote, early layers seekhte nahi (fix: ReLU, BatchNorm, residual connections, LSTM)
- **Exploding gradient:** gradients bahut bade (fix: gradient clipping)

---

[Previous: 2. Introduction to Deep Learning](02-introduction-to-deep-learning.md) | [Home](../README.md) | [Next: 4. Computer Vision](04-computer-vision.md)
