import cv2
from ultralytics import YOLO
import time

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Colors for classes
COLORS = {
    "With Helmet": (0, 255, 0),      # Green
    "Without Helmet": (0, 0, 255)    # Red
}

def get_input_source(source=None, droidcam_ip=None):
    if isinstance(source, str) and source.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        print("Using Image:", source)
        img = cv2.imread(source)
        return "image", img
    if droidcam_ip:
        print("Using DroidCam at:", droidcam_ip)
        cap = cv2.VideoCapture(f"http://{droidcam_ip}/video")
    else:
        print("Using default webcam")
        cap = cv2.VideoCapture(0)
    return "camera", cap

def detect_helmet(image_or_frame, showText=True):
    if isinstance(image_or_frame, str):
        img = cv2.imread(image_or_frame)
        results = model(image_or_frame)
    else:
        img = image_or_frame.copy()
        results = model(img)

    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            color = COLORS.get(label, (255, 255, 0))

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            if showText:
                text = f"{label.upper()} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(img, (x1, y2 + 5), (x1 + w + 6, y2 + h + 15), color, -1)
                cv2.putText(img, text, (x1 + 3, y2 + h + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return img, detections

# DroidCam IP & image path
droidcam_ip = ""  # Replace with your IP
image_path = None

mode, data = get_input_source(source=image_path, droidcam_ip=droidcam_ip)

if mode == "image":
    output, dets = detect_helmet(data, showText=True)
    for d in dets:
        print(f"{d['label']} - {d['confidence']:.2f} - {d['bbox']}")
    cv2.imshow("Result", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif mode == "camera":
    cap = data
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame from camera, retrying...")
            continue

        output, dets = detect_helmet(frame, showText=True)

        # Print all detections
        for d in dets:
            print(f"{d['label']} - {d['confidence']:.2f} - {d['bbox']}")

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Live Helmet Detection", output)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
