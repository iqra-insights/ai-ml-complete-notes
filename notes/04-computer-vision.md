[Previous: 3. Artificial Neural Network Deep Dive](03-artificial-neural-network-deep-dive.md) | [Home](../README.md) | [Next: 5. Computer Vision with OpenCV: Images, Video, OCR, MediaPipe](05-computer-vision-with-opencv-images-video-ocr-med.md)

---

## 4. Computer Vision

Computer Vision (CV) = machines ko images/videos 'dekhna' aur samajhna sikhana.

### 4.1 Tasks ka difference

```
Classification:  'Isme billi hai'            (1 label per image)
Localisation:    'Billi yahan hai' + box      (1 object)
Object Detection:'2 billi, 1 kutta' + boxes   (multiple objects)
Semantic Seg.:   har pixel ka class label     (all cats = same colour)
Instance Seg.:   har object alag mask         (cat1, cat2 alag)
```

### 4.2 Object Detection

Output: **bounding box (x, y, w, h)** + **class** + **confidence score**.

**Key concepts**

- **IoU (Intersection over Union)** = overlap area / union area. IoU > 0.5 usually correct detection.
- **NMS (Non-Max Suppression):** same object ke multiple boxes mein se best rakho, baaki hatao.
- **Anchor boxes:** predefined shapes jinse boxes predict hote hain.
- **mAP (mean Average Precision):** detection ka main metric.

**Two families**

| Two-stage (accurate, slower) | One-stage (fast, real-time) |
| --- | --- |
| R-CNN, Fast R-CNN, Faster R-CNN | YOLO, SSD, RetinaNet |
| Pehle regions propose, phir classify | Ek pass mein box + class |

### 4.3 Region Proposal Techniques

Sawal: image mein object kahan ho sakta hai? Har jagah check karna (sliding window) bahut slow hai.

| Method | Idea |
| --- | --- |
| Sliding window | Har location + size try karo (very slow) |
| Selective Search | Similar colour/texture/size wale regions merge karke \~2000 proposals (R-CNN) |
| RPN (Region Proposal Network) | CNN khud proposals seekhta hai (Faster R-CNN) |

**Evolution:**

<p align="center"><img src="../images/06-object-detection-evolution.png" alt="object detection evolution · R-CNN to YOLO" width="800"><br><sub><i>object detection evolution · R-CNN to YOLO</i></sub></p>

### 4.4 Semantic Segmentation

Har pixel ko class label dena. Output = image jitna bada mask.

<p align="center"><img src="../images/07-segmentation-encoder-decoder.png" alt="segmentation encoder-decoder with skip connections" width="800"><br><sub><i>segmentation encoder-decoder with skip connections</i></sub></p>

- **U-Net:** encoder-decoder + skip connections (medical imaging mein popular)
- **FCN:** fully convolutional network
- **DeepLab:** atrous (dilated) convolution
- **Mask R-CNN:** instance segmentation
- **Metrics:** IoU / Dice coefficient / pixel accuracy

---

[Previous: 3. Artificial Neural Network Deep Dive](03-artificial-neural-network-deep-dive.md) | [Home](../README.md) | [Next: 5. Computer Vision with OpenCV: Images, Video, OCR, MediaPipe](05-computer-vision-with-opencv-images-video-ocr-med.md)
