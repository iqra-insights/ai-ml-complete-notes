# Reference snippets from notes/02-introduction-to-deep-learning.md
# These are study snippets, run them piece by piece with your own data (X, y, df, etc. are placeholders).

# ---- Snippet 1 ----
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

