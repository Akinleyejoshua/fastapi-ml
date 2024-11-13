from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import cv2
import numpy as np
import io
from PIL import Image
import base64
import matplotlib.pyplot as plt

app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', ''],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"msg": "API WORKING"}

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_json(message)
            
manager = ConnectionManager()

def process_img_for_prediction(img_np, img_size):
    img_arr = tf.image.convert_image_dtype(img_np, tf.float32)
    img_arr = tf.image.resize(img_arr, size=[img_size, img_size])
    img_arr = tf.expand_dims(img_arr, 0)
    return img_arr

def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model("../model.h5")

face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def make_prediction(model, img_np, img_size):

    img = process_img_for_prediction(img_np, img_size)
    pred = model.predict(img, verbose=1)
    return classes[np.argmax(pred)], np.max(pred) * 100

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "analyse-face":
                data_str = data["img"]
                point = data_str.find(",")
                base64_str = data_str[point:]
                image = base64.b64decode(base64_str)
                img = Image.open(io.BytesIO(image))

                if img.mode != "RGB":
                    img = img.convert("RGB")

                image_np = np.array(img)
                img_tf = tf.constant(img)
                
                gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                # Process each face
                output = []
                for (x, y, w, h) in faces:
                    face = img_tf[y:y+h, x:x+w]
                    face = process_img_for_prediction(face, 84)
                    pred_ = model.predict(face, verbose=0)
                    pred = classes[np.argmax(pred_)]
                    score = np.max(pred_) * 100

                    output.append({"x": f"{x}", 
                                   "y": f"{y}", 
                                   "w": f"{w}", 
                                   "h": f"{h}", 
                                   "pred": f"{pred}", 
                                   "score": f"{score}"})

                await manager.send_personal_message({"type": 
                                                     "analyse-face", 
                                                     "output": output}, websocket)
                    
            if data["type"] == "greet":
                await manager.send_personal_message(data, websocket)
                await manager.broadcast({"type": "info", "msg": f"#{client_id} said {data['msg']}"})


    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast({"type": "info", "msg": f"#{client_id} disconnected"})


# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, reload=True)