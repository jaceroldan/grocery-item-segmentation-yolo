import gradio as gr
import asyncio
import time
import cv2
from ultralytics import YOLO
import numpy as np
import os

# Ensure cache directories exist
os.environ["GRADIO_TEMP_DIR"] = "./gradio_cache"
os.environ["GRADIO_CACHE_DIR"] = "./gradio_cache"
os.makedirs("./gradio_cache", exist_ok=True)

# Load the YOLO model
model = YOLO('runs/segment/train9/weights/best.pt')

async def process_frame_webcam(input_image):
    # Flip the input image horizontally (mirroring for webcam)
    input_image = cv2.flip(input_image, 1)

    # Convert the input image from BGR to RGB for the YOLO model
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Run YOLO model on the frame
    results = model(input_image)

    # Check if `results` is a list, take the first element if so
    result = results[0] if isinstance(results, list) else results

    # Render the segmentation result on the frame
    segmented_frame = result.plot()  # `plot()` overlays results on the image

    return segmented_frame  # Return the NumPy array

def gradio_infer(input_image):
    start_time = time.time()

    # Process the input frame using asyncio.run for async function
    segmented_frame = asyncio.run(process_frame_webcam(input_image))

    # Convert the segmented frame to RGB for Gradio display
    segmented_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2RGB)

    # Calculate FPS
    fps = 1 / (time.time() - start_time) / 10

    # Display FPS on the frame
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
    inputs=gr.Image(label="input", sources="webcam", streaming=True),
    outputs=gr.Image(),
    live=True,
    title="Real-Time Grocery Segmentation",
    description="A YOLO-based model for grocery segmentation using your webcam."
)

if __name__ == "__main__":
    webcam_interface.launch(share=True)
