import cv2
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from starlette.responses import StreamingResponse
import numpy as np
import io

app = FastAPI()
model = YOLO('runs/segment/train/weights/best.pt')  # Load your trained segmentation model

@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    # Read the uploaded frame
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Perform inference
    results = model(frame)

    # Check if `results` is a list, take the first element if so
    if isinstance(results, list):
        result = results[0]
    else:
        result = results

    # Render the segmentation result on the frame
    segmented_frame = result.plot()  # `plot()` is the method to overlay results on the image


    # Encode the segmented frame for sending back
    _, buffer = cv2.imencode('.jpg', segmented_frame)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5678)
