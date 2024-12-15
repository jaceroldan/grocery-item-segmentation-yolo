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
model = YOLO('runs/segment/train20/weights/best.pt')

async def process_frame_webcam(input_image):
    # Because my laptop cam is mirrored
    start_time = time.time()
    input_image = cv2.flip(input_image, 1)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) # Reverse color channels

    results = model(input_image)
    result = results[0] if isinstance(results, list) else results

    segmented_frame = result.plot()  # `plot()` overlays results on the image
    segmented_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2RGB) # Reverse color channels
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
with gr.Blocks() as demo:
    with gr.Group(elem_classes="my-group"):
        
        input_img = gr.Image(sources=['webcam'], 
                            type="numpy", streaming=True, label="Webcam",
                            # mirror_webcam=False,
                            )
        input_img.stream(
            fn=process_frame_webcam,
            inputs=[input_img],
            outputs=input_img,
            time_limit=10, 
            concurrency_limit=10,


            stream_every=0.6, # BEST SO FAR IN CONDO
                                # # use this for now, in order to still have a decent FPS
                                # 1.6 fps
            # stream_every=0.075, # BEST SO FAR IN EDUROAM, 0.033 still laggy
            

            
        )

# demo.launch(share=True)

# demo.queue()
demo.launch(share=True)