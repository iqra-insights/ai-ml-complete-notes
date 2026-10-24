[Previous: 1. Understanding the World of AI](01-understanding-the-world-of-ai.md) | [Home](../README.md) | [Next: 3. Artificial Neural Network Deep Dive](03-artificial-neural-network-deep-dive.md)

---

## 2. Introduction to Deep Learning

### 2.1 What is Deep Learning?

DL = neural networks jisme **multiple hidden layers** hote hain. Ye raw data (pixels, text) se khud features seekhta hai.

**Why now?** Big data + GPUs + better algorithms.

### 2.2 Perceptron Model

Perceptron = sabse simple neural network (single neuron). Biological neuron se inspired.

<p align="center"><img src="../images/03-perceptron.png" alt="perceptron · inputs, weights, sum, activation" width="800"><br><sub><i>perceptron · inputs, weights, sum, activation</i></sub></p>

**Formula:** `z = w1x1 + w2x2 + ... + wnxn + b`, then `y = f(z)`

**Learning rule:** `w_new = w_old + lr * (y_true - y_pred) * x`

**Limitation:** Single perceptron sirf **linearly separable** problems solve karta hai. XOR nahi kar sakta. Isliye **Multi-Layer Perceptron (MLP)** aaya: hidden layers + non-linear activation.

<p align="center"><img src="../images/04-mlp-network.png" alt="multi-layer perceptron · 3-4-4-2 network" width="800"><br><sub><i>multi-layer perceptron · 3-4-4-2 network</i></sub></p>

### 2.3 TensorFlow vs PyTorch vs Keras

| Feature | TensorFlow | PyTorch | Keras |
| --- | --- | --- | --- |
| By | Google | Meta (Facebook) | Now part of TensorFlow (multi-backend in Keras 3) |
| Level | Low + high | Low-level, flexible | High-level API |
| Graph | Static + eager | Dynamic (define-by-run) | Abstraction |
| Best for | Production, deployment | Research, debugging | Beginners, quick prototyping |

**Keras example:**

```python
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
```

### 2.4 Types of Neural Networks

| Network | Best for | Key idea |
| --- | --- | --- |
| ANN / MLP | Tabular data | Fully connected layers |
| CNN | Images, video | Convolution filters |
| RNN | Sequences | Hidden state memory |
| LSTM / GRU | Long sequences | Gates control memory |
| Autoencoder | Compression, anomaly | Encoder + decoder |
| GAN | Generation | Generator vs discriminator |
| Transformer | NLP, LLMs | Self-attention |

---

[Previous: 1. Understanding the World of AI](01-understanding-the-world-of-ai.md) | [Home](../README.md) | [Next: 3. Artificial Neural Network Deep Dive](03-artificial-neural-network-deep-dive.md)
