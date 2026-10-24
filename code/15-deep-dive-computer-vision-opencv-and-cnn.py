# Reference snippets from notes/15-deep-dive-computer-vision-opencv-and-cnn.md
# These are study snippets, run them piece by piece with your own data (X, y, df, etc. are placeholders).

# ---- Snippet 1 ----
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))      # green range
result = cv2.bitwise_and(img, img, mask=mask)

# ---- Snippet 2 ----
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    if cv2.contourArea(c) > 500:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# ---- Snippet 3 ----
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# ---- Snippet 4 ----
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
text = pytesseract.image_to_string(th, config='--oem 3 --psm 6')   # psm 6 = single block of text

# ---- Snippet 5 ----
import numpy as np
def angle(a, b, c):                       # points as (x, y); b = joint
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = abs(np.degrees(rad))
    return 360 - deg if deg > 180 else deg
# bicep curl: shoulder, elbow, wrist; angle < 40 = up, > 160 = down -> count rep on down->up change

# ---- Snippet 6 ----
aug = keras.Sequential([keras.layers.RandomFlip('horizontal'), keras.layers.RandomRotation(0.1), keras.layers.RandomZoom(0.1)])

# ---- Snippet 7 ----
base = keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base.trainable = False                    # pehle freeze
model = keras.Sequential([base, keras.layers.GlobalAveragePooling2D(), keras.layers.Dropout(0.3), keras.layers.Dense(5, activation='softmax')])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# phir top layers unfreeze karke lr 1e-5 par fine-tune karo

