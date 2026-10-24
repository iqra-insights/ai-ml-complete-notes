# Reference snippets from notes/14-deep-dive-ml-basics-perceptron-and-ann.md
# These are study snippets, run them piece by piece with your own data (X, y, df, etc. are placeholders).

# ---- Snippet 1 ----
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

