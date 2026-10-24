# Reference snippets from notes/05-computer-vision-with-opencv-images-video-ocr-med.md
# These are study snippets, run them piece by piece with your own data (X, y, df, etc. are placeholders).

# ---- Snippet 1 ----
import cv2
img = cv2.imread('photo.jpg')                  # read (BGR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # grayscale
resized = cv2.resize(img, (300, 300))          # resize
blur = cv2.GaussianBlur(img, (5, 5), 0)        # blur / denoise
edges = cv2.Canny(gray, 100, 200)              # edge detection
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 2)
cv2.imwrite('out.jpg', img)

# ---- Snippet 2 ----
cap = cv2.VideoCapture(0)          # 0 = webcam, ya 'video.mp4'
while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Live', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()

# ---- Snippet 3 ----
import cv2, pytesseract
img = cv2.imread('doc.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
text = pytesseract.image_to_string(th, lang='eng')
boxes = pytesseract.image_to_data(th, output_type=pytesseract.Output.DICT)  # word boxes

# ---- Snippet 4 ----
import mediapipe as mp, cv2
mp_pose = mp.solutions.pose
with mp_pose.Pose(min_detection_confidence=0.5) as pose:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # MediaPipe needs RGB
    result = pose.process(rgb)
    # result.pose_landmarks -> x, y, z, visibility

