import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")



# Colors for classes
COLORS = {
    "With Helmet": (0, 255, 0),      # Green
    "Without Helmet": (0, 0, 255)    # Red
}


def get_input_source(source=None, droidcam_ip=None):
    """
    Automatically select input source.
    
    Args:
        source (str or None):
            - Image path → uses image
            - None → uses camera (DroidCam if droidcam_ip provided)
        droidcam_ip (str or None):
            - IP address of DroidCam (http://<IP>:<PORT>/video)
    
    Returns:
        mode ("image" or "camera"), data (image or VideoCapture)
    """
    # If image path is provided
    if isinstance(source, str) and source.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        print("Using Image:", source)
        img = cv2.imread(source)
        return "image", img

    # Otherwise use camera
    if droidcam_ip:
        print("Using DroidCam at:", droidcam_ip)
        cap = cv2.VideoCapture(f"http://{droidcam_ip}/video")
    else:
        print("Using default webcam")
        cap = cv2.VideoCapture(0)
    
    return "camera", cap


def detect_helmet(image_or_frame, showText=True):
    """
    Detect helmet / no-helmet in an image or frame

    Args:
        image_or_frame (str or numpy array): image path OR image frame
        showText (bool): If True → show label text, else only boxes

    Returns:
        output_image
        detections
    """

    # If input is a path
    if isinstance(image_or_frame, str):
        img_orig = cv2.imread(image_or_frame)
        img_input = cv2.resize(img_orig, (416, 416))
        results = model(img_input)
    else:
        img_orig = image_or_frame.copy()
        img_input = cv2.resize(img_orig, (416, 416))
        results = model(img_input)

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
            print("yeeee",conf, label)
            cv2.rectangle(img_input, (x1, y1), (x2, y2), color, 2)

            if showText:
                text = f"{label.upper()} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                cv2.rectangle(img_input, (x1, y2 + 5), (x1 + w + 6, y2 + h + 15), color, -1)
                cv2.putText(img_input, text, (x1 + 3, y2 + h + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    return img_input, detections


# Input image
#image_path = r"D:\from scrach\test\images\BikesHelmets269_png_jpg.rf.da527144e8b088ab97d197544ed0b87a.jpg"
#image_path = r"D:\from scrach\test\images\BikesHelmets739_png_jpg.rf.63df6a974579d37a25c27ccff6900fb5.jpg"
image_path = r"D:\from scrach\test\images\without.jpg"
droidcam_ip = "192.168.0.117:4747"  # Replace with your phone IP & port, or None
# image_path = None

mode, data = get_input_source(source=image_path, droidcam_ip=droidcam_ip)

if mode == "image":
    output, dets = detect_helmet(data, showText=True)
    cv2.imshow("Result", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif mode == "camera":
    cap = data

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera")
            break

        output, dets = detect_helmet(frame, showText=True)

        # Print violations
        for d in dets:
            if d["label"] == "Without Helmet":
                print("NO HELMET:", d)

        cv2.imshow("Live Helmet Detection", output)

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
