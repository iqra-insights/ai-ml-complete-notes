[Previous: 5. Computer Vision with OpenCV: Images, Video, OCR, MediaPipe](05-computer-vision-with-opencv-images-video-ocr-med.md) | [Home](../README.md) | [Next: 7. NLP: Sentiment Analysis on Amazon Reviews](07-nlp-sentiment-analysis-on-amazon-reviews.md)

---

## 6. Convolutional Neural Network (CNN)

### 6.1 Why CNN?

Image par ANN use karo to parameters explode ho jate hain (224x224x3 = 150k inputs) aur spatial structure khatam. CNN **local patterns** seekhta hai aur **weights share** karta hai.

### 6.2 Architecture

<p align="center"><img src="../images/08-cnn-architecture.png" alt="CNN architecture · feature extraction then classification" width="800"><br><sub><i>CNN architecture · feature extraction then classification</i></sub></p>

**Kernel / Filter:** chhota matrix (3x3, 5x5) jo image par slide karke feature map banata hai. Early layers = edges, deeper = shapes, objects.

```
Image patch      Kernel        Feature map value
[1 2 0]          [1 0 -1]
[0 1 3]    *     [1 0 -1]   =   sum of element-wise products
[2 1 0]          [1 0 -1]
```

**Stride:** filter ek baar mein kitne pixel jump kare. Bada stride = chhota output.

**Padding:** border par zeros add karna.

- Valid = no padding (output shrinks)
- Same = padding so output size = input size

**Output size formula:** `O = (N - F + 2P) / S + 1` Example: N=32, F=3, P=0, S=1 gives O = 30.

**Pooling:** downsampling, parameters kam, thodi translation invariance.

- Max pooling: window ka maximum (strong features)
- Average pooling: window ka average (smooth)

```
Max Pool 2x2:  [1 3 | 2 4]        [3 | 4]
               [5 6 | 7 8]   -->  [6 | 8]
```

**Activation:** ReLU conv ke baad. **Flatten:** 2D maps ko 1D vector. **Dense + Softmax:** final classification.

```python
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])
```

**Famous CNNs:** LeNet, AlexNet, VGG, ResNet (skip connections), Inception, MobileNet, EfficientNet. **Transfer learning:** pretrained model (ImageNet) ko apne data par fine-tune karo.

---

[Previous: 5. Computer Vision with OpenCV: Images, Video, OCR, MediaPipe](05-computer-vision-with-opencv-images-video-ocr-med.md) | [Home](../README.md) | [Next: 7. NLP: Sentiment Analysis on Amazon Reviews](07-nlp-sentiment-analysis-on-amazon-reviews.md)
