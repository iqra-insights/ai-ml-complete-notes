[Previous: 4. Computer Vision](04-computer-vision.md) | [Home](../README.md) | [Next: 6. Convolutional Neural Network (CNN)](06-convolutional-neural-network-cnn.md)

---

## 5. Computer Vision with OpenCV: Images, Video, OCR, MediaPipe

### 5.1 Working with Images

Image = pixels ka matrix. Grayscale = 2D (H x W), Colour = 3D (H x W x 3). **OpenCV BGR order use karta hai, RGB nahi.**

```python
import cv2
img = cv2.imread('photo.jpg')                  # read (BGR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # grayscale
resized = cv2.resize(img, (300, 300))          # resize
blur = cv2.GaussianBlur(img, (5, 5), 0)        # blur / denoise
edges = cv2.Canny(gray, 100, 200)              # edge detection
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 2)
cv2.imwrite('out.jpg', img)
```

**Common operations:** crop, rotate, flip, thresholding, morphology (erode/dilate), contours, histogram equalisation.

### 5.2 Working with Videos

Video = frames ka sequence. Loop mein frame by frame process karo.

```python
cap = cv2.VideoCapture(0)          # 0 = webcam, ya 'video.mp4'
while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Live', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()
```

Save video: `cv2.VideoWriter` + codec (fourcc). Use cases: motion detection, tracking, surveillance.

### 5.3 OCR (Tesseract)

OCR = image se text nikalna.

```
Image --> Preprocess (gray, threshold, denoise) --> Text detection --> Recognition --> Text
```

```python
import cv2, pytesseract
img = cv2.imread('doc.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
text = pytesseract.image_to_string(th, lang='eng')
boxes = pytesseract.image_to_data(th, output_type=pytesseract.Output.DICT)  # word boxes
```

**Accuracy tips:** high resolution, good contrast, deskew, noise removal. Tesseract engine install karna padta hai (pytesseract sirf wrapper hai). Alternatives: EasyOCR, PaddleOCR.

### 5.4 MediaPipe (FaceMesh, Pose Detection)

Google ka ready-made, real-time ML framework (lightweight, CPU par bhi chalta hai).

| Solution | Output |
| --- | --- |
| Face Mesh | 468 3D face landmarks |
| Pose | 33 body landmarks |
| Hands | 21 landmarks per hand |

```python
import mediapipe as mp, cv2
mp_pose = mp.solutions.pose
with mp_pose.Pose(min_detection_confidence=0.5) as pose:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # MediaPipe needs RGB
    result = pose.process(rgb)
    # result.pose_landmarks -> x, y, z, visibility
```

**Uses:** AR filters, fitness rep counter, yoga posture check, gesture control, drowsiness detection.

---

[Previous: 4. Computer Vision](04-computer-vision.md) | [Home](../README.md) | [Next: 6. Convolutional Neural Network (CNN)](06-convolutional-neural-network-cnn.md)
