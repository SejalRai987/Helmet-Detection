import cv2
from ultralytics import YOLO

# ==============================
# LOAD MODEL
# ==============================
model = YOLO(r"D:\from scrach\best.pt")




# ==============================
# CONFIGURATION (TUNE HERE)
# ==============================
IMG_SIZE = 832        # Higher = better small object detection
CONF_THRESHOLD = 0.35 # Lower = detect more people
IOU_THRESHOLD = 0.5   # NMS control

# Class colors
COLORS = {
    "With Helmet": (0, 255, 0),
    "Without Helmet": (0, 0, 255)
}


# ==============================
# INPUT SOURCE HANDLER
# ==============================
def get_input_source(source=None, droidcam_ip=None):

    # If image path provided
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


# ==============================
# HELMET DETECTION FUNCTION
# ==============================
def detect_helmet(frame, showText=True):

    # YOLO handles resizing internally (important!)
    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD
    )

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

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if showText:
                text = f"{label.upper()} {conf:.2f}"

                (w, h), _ = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )

                # Draw filled rectangle for text background
                cv2.rectangle(
                    frame,
                    (x1, y1 - h - 10),
                    (x1 + w + 6, y1),
                    color,
                    -1
                )

                # Put text
                cv2.putText(
                    frame,
                    text,
                    (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

    return frame, detections


# ==============================
# MAIN EXECUTION
# ==============================

# Choose ONE:
image_path = r"D:\from scrach\BikesHelmets0_png_jpg.rf.6b5704cd71c95602bcb6d97a40fb2ac4.jpg"

#image_path = r"D:\from scrach\test\images\without.jpg"
droidcam_ip = None  # Example: "192.168.0.117:4747"

mode, data = get_input_source(source=image_path, droidcam_ip=droidcam_ip)

# ------------------------------
# IMAGE MODE
# ------------------------------
if mode == "image":

    output, dets = detect_helmet(data, showText=True)

    print(f"\nTotal detections: {len(dets)}")

    for d in dets:
        if d["label"] == "Without Helmet":
            print("NO HELMET:", d)

    cv2.imshow("Helmet Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------------------------------
# CAMERA MODE
# ------------------------------
elif mode == "camera":

    cap = data

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        output, dets = detect_helmet(frame, showText=True)

        for d in dets:
            if d["label"] == "Without Helmet":
                print("NO HELMET:", d)

        cv2.imshow("Live Helmet Detection", output)

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


#############################
# import cv2
# from ultralytics import YOLO

# # ==============================
# # LOAD MODEL
# # ==============================
# model = YOLO(r"D:\from scrach\best.pt")




# # ==============================
# # CONFIGURATION (TUNE HERE)
# # ==============================
# IMG_SIZE = 832        # Higher = better small object detection
# CONF_THRESHOLD = 0.35 # Lower = detect more people
# IOU_THRESHOLD = 0.5   # NMS control

# # Class colors
# COLORS = {
#     "With Helmet": (0, 255, 0),
#     "Without Helmet": (0, 0, 255)
# }


# # ==============================
# # INPUT SOURCE HANDLER
# # ==============================
# def get_input_source(source=None, droidcam_ip=None):

#     # If image path provided
#     if isinstance(source, str) and source.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
#         print("Using Image:", source)
#         img = cv2.imread(source)
#         return "image", img

#     # Otherwise use camera
#     if droidcam_ip:
#         print("Using DroidCam at:", droidcam_ip)
#         cap = cv2.VideoCapture(f"http://{droidcam_ip}/video")
#     else:
#         print("Using default webcam")
#         cap = cv2.VideoCapture(0)

#     return "camera", cap


# # ==============================
# # HELMET DETECTION FUNCTION
# # ==============================
# def detect_helmet(frame, showText=True):

#     # YOLO handles resizing internally (important!)
#     results = model(
#         frame,
#         imgsz=IMG_SIZE,
#         conf=CONF_THRESHOLD,
#         iou=IOU_THRESHOLD
#     )

#     detections = []

#     for result in results:
#         for box in result.boxes:

#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = float(box.conf[0])
#             cls = int(box.cls[0])

#             label = model.names[cls]
#             color = COLORS.get(label, (255, 255, 0))

#             detections.append({
#                 "label": label,
#                 "confidence": conf,
#                 "bbox": [x1, y1, x2, y2]
#             })

#             # Draw bounding box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#             if showText:
#                 text = f"{label.upper()} {conf:.2f}"

#                 (w, h), _ = cv2.getTextSize(
#                     text,
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     2
#                 )

#                 # Draw filled rectangle for text background
#                 cv2.rectangle(
#                     frame,
#                     (x1, y1 - h - 10),
#                     (x1 + w + 6, y1),
#                     color,
#                     -1
#                 )

#                 # Put text
#                 cv2.putText(
#                     frame,
#                     text,
#                     (x1 + 3, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     (255, 255, 255),
#                     2
#                 )

#     return frame, detections


# # ==============================
# # MAIN EXECUTION
# # ==============================

# # Choose ONE:
# image_path = None#r"D:\from scrach\BikesHelmets0_png_jpg.rf.6b5704cd71c95602bcb6d97a40fb2ac4.jpg"

# #image_path = r"D:\from scrach\test\images\without.jpg"
# droidcam_ip =  "10.49.192.250"

# mode, data = get_input_source(source=image_path, droidcam_ip=droidcam_ip)

# # ------------------------------
# # IMAGE MODE
# # ------------------------------
# if mode == "image":

#     output, dets = detect_helmet(data, showText=True)

#     print(f"\nTotal detections: {len(dets)}")

#     for d in dets:
#         if d["label"] == "Without Helmet":
#             print("NO HELMET:", d)

#     cv2.imshow("Helmet Detection", output)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # ------------------------------
# # CAMERA MODE
# # ------------------------------
# elif mode == "camera":

#     cap = data

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         output, dets = detect_helmet(frame, showText=True)

    #     for d in dets:
    #         if d["label"] == "Without Helmet":
    #             print("NO HELMET:", d)

    #     cv2.imshow("Live Helmet Detection", output)

    #     # Press ESC to exit
    #     if cv2.waitKey(1) & 0xFF == 27:
    #         break

    # cap.release()
    # cv2.destroyAllWindows()


