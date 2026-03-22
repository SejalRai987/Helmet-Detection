from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os
import cv2
from ultralytics import YOLO

app = FastAPI()

# Create folders
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load YOLO model
model = YOLO("best.pt")   # your trained model

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------- IMAGE DETECTION ----------------
@app.post("/detect-image")
async def detect_image(request: Request, file: UploadFile = File(...)):

    file_path = f"static/uploads/{file.filename}"
    result_path = f"static/results/result_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(file_path)
    annotated_frame = results[0].plot()

    cv2.imwrite(result_path, annotated_frame)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result_image": result_path
    })


# ---------------- VIDEO DETECTION ----------------
@app.post("/detect-video")
async def detect_video(request: Request, file: UploadFile = File(...)):

    file_path = f"static/uploads/{file.filename}"
    result_path = f"static/results/result_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(file_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result_video": result_path
    })


# ---------------- LIVE CAMERA ----------------
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model(frame)
            annotated_frame = results[0].plot()

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.get("/live")
def live_camera():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")