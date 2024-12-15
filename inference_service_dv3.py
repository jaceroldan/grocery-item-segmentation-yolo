import gradio as gr
import asyncio
import time
import cv2
from ultralytics import YOLO
import os

# Ensure cache directories exist
os.environ["GRADIO_TEMP_DIR"] = "./gradio_cache"
os.environ["GRADIO_CACHE_DIR"] = "./gradio_cache"
os.makedirs("./gradio_cache", exist_ok=True)

# Load the YOLO model
model = YOLO('runs/segment/train13/weights/best.onnx')

async def process_frame_webcam(input_image):
    # Because my laptop cam is mirrored
    input_image = cv2.flip(input_image, 1)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) # Reverse color channels

    results = model(input_image)
    result = results[0] if isinstance(results, list) else results

    segmented_frame = result.plot()  # `plot()` overlays results on the image
    return segmented_frame

def gradio_infer(input_image):
    start_time = time.time()
    segmented_frame = asyncio.run(process_frame_webcam(input_image))
    segmented_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2RGB)

    fps = 1 / (time.time() - start_time) / 10

    height, width, _ = segmented_frame.shape
    cv2.putText(
        segmented_frame,
        f"FPS: {fps:.2f}",
        (10, height - 10),  # Position FPS text at the bottom-left
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return segmented_frame

# Define Gradio interface
webcam_interface = gr.Interface(
    fn=gradio_infer,
    inputs=gr.Image(label="input", sources="webcam", streaming=True, height=480, width=640),
    outputs=gr.Image(),
    live=True,
    title="Real-Time Grocery Segmentation",
    description="A YOLO-based model for grocery segmentation using your webcam.",
    stream_every=0.6,
)

if __name__ == "__main__":
    webcam_interface.launch(share=True)
