[Previous: 14. Deep Dive: ML Basics, Perceptron and ANN](14-deep-dive-ml-basics-perceptron-and-ann.md) | [Home](../README.md) | [Next: 16. Deep Dive: NLP, Sentiment, Spam, Translation and LLMs](16-deep-dive-nlp-sentiment-spam-translation-and-llm.md)

---

## 15. Deep Dive: Computer Vision, OpenCV and CNN

### 15.1 Image fundamentals

- Pixel value 0-255 (8-bit). Grayscale = 1 channel, colour = 3 channels (OpenCV: BGR).
- **HSV** colour space (Hue, Saturation, Value) colour-based detection ke liye RGB se behtar hai.
- Neural networks ke liye images ko 0-1 range mein normalise karo (`img / 255.0`).

### 15.2 Object detection in more detail

```math
IoU=\frac{\text{Area of overlap}}{\text{Area of union}}
```

**NMS algorithm:** (1) boxes ko confidence se sort karo, (2) sabse high box rakho, (3) uske saath IoU > threshold wale boxes hatao, (4) bache hue par repeat.

**YOLO idea:** image ko S x S grid mein baantta hai. Har cell B boxes aur C class probabilities predict karta hai. Output tensor: `S x S x (B*5 + C)` (har box: x, y, w, h, confidence).

**Metrics:** Precision, Recall, **AP** (precision-recall curve ke neeche area), **mAP** (sab classes ka average AP, e.g. mAP@0.5).

### 15.3 Segmentation in more detail

```math
Dice=\frac{2|A\cap B|}{|A|+|B|}
```

- **U-Net:** encoder (downsample) + decoder (upsample) + **skip connections** jo fine details wapas laate hain. Medical images mein bahut popular.
- **Semantic:** class per pixel. **Instance:** har object alag (Mask R-CNN). **Panoptic:** dono ka mix.

### 15.4 OpenCV recipes

**Colour detection (HSV mask):**

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))      # green range
result = cv2.bitwise_and(img, img, mask=mask)
```

**Contours:**

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    if cv2.contourArea(c) > 500:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
```

**Face detection (Haar cascade):**

```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
```

**OCR preprocessing (low quality scans):**

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
text = pytesseract.image_to_string(th, config='--oem 3 --psm 6')   # psm 6 = single block of text
```

**Tesseract page segmentation (`--psm`):** 3 auto, 6 uniform block, 7 single line, 11 sparse text.

### 15.5 MediaPipe: joint angle (rep counter / posture)

```python
import numpy as np
def angle(a, b, c):                       # points as (x, y); b = joint
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = abs(np.degrees(rad))
    return 360 - deg if deg > 180 else deg
# bicep curl: shoulder, elbow, wrist; angle < 40 = up, > 160 = down -> count rep on down->up change
```

### 15.6 CNN in more detail

**Parameters of a conv layer:** `(F x F x C_in + 1) x C_out`. Example: 3x3 kernel, 3 input channels, 32 filters = (3 x 3 x 3 + 1) x 32 = **896 parameters**.

**Output size worked example:** input 64x64, kernel 3, padding 0, stride 1 gives (64-3)/1+1 = 62. After 2x2 max pool gives 31.

| Idea | Why it matters |
| --- | --- |
| Weight sharing | Ek filter poori image par slide, parameters bahut kam |
| Receptive field | Deeper neuron image ka bada hissa 'dekhta' hai |
| 1x1 convolution | Channels kam/zyada karna, kam compute |
| Padding same | Spatial size preserve |
| Global Average Pooling | Flatten ki jagah, parameters kam, overfitting kam |

**Famous CNNs**

| Model | Key innovation |
| --- | --- |
| LeNet-5 | Pehla practical CNN (digits) |
| AlexNet | ReLU + Dropout + GPU, ImageNet 2012 jeeta |
| VGG | Sirf 3x3 filters, bahut deep |
| Inception | Multi-size filters parallel |
| ResNet | **Skip connections**, 100+ layers train ho sakte hain |
| MobileNet | Depthwise separable conv, mobile ke liye halka |
| EfficientNet | Depth, width, resolution ek saath scale |

**Data augmentation (Keras layers):**

```python
aug = keras.Sequential([keras.layers.RandomFlip('horizontal'), keras.layers.RandomRotation(0.1), keras.layers.RandomZoom(0.1)])
```

**Transfer learning (MobileNetV2):**

```python
base = keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base.trainable = False                    # pehle freeze
model = keras.Sequential([base, keras.layers.GlobalAveragePooling2D(), keras.layers.Dropout(0.3), keras.layers.Dense(5, activation='softmax')])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# phir top layers unfreeze karke lr 1e-5 par fine-tune karo
```

---

[Previous: 14. Deep Dive: ML Basics, Perceptron and ANN](14-deep-dive-ml-basics-perceptron-and-ann.md) | [Home](../README.md) | [Next: 16. Deep Dive: NLP, Sentiment, Spam, Translation and LLMs](16-deep-dive-nlp-sentiment-spam-translation-and-llm.md)
