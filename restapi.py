from fastapi import FastAPI, WebSocket, UploadFile, File, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import base64

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# Allow frontend to connect (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = load_model("ctscan_model.h5")

def predict_image_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "sakit" if prediction > 0.5 else "normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    print(prediction)
    return label, confidence


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            image_data = base64.b64decode(data.split(",")[1])  # Remove data:image/jpeg;base64,
            label, confidence = predict_image_from_bytes(image_data)
            await websocket.send_json({ "label": label, "confidence": round(float(confidence), 3)})
        except Exception as e:
            await websocket.send_json({"error": str(e)})
            break
