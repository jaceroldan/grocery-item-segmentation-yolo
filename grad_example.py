import gradio as gr
import numpy as np
import cv2


def transform_cv2(frame, transform, save_path='input_image.jpg'):
    if transform == "cartoon":
        # prepare color
        img_color = cv2.pyrDown(cv2.pyrDown(frame))
        for _ in range(6):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        img_color = cv2.pyrUp(cv2.pyrUp(img_color))

        # prepare edges
        img_edges = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        img_edges = cv2.adaptiveThreshold(
            cv2.medianBlur(img_edges, 7),
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            2,
        )
        img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

        # combine color and edges
        img = cv2.bitwise_and(img_color, img_edges)

    elif transform == "edges":
        # perform edge detection
        img = cv2.cvtColor(cv2.Canny(frame, 100, 200), cv2.COLOR_GRAY2BGR)

    else:
        img = np.flipud(frame)

    # Save the image if save_path is provided
    if save_path:
        cv2.imwrite(save_path, img)

    return img


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            transform = gr.Dropdown(choices=["cartoon", "edges", "flip"],
                                    value="flip", label="Transformation")
            input_img = gr.Image(sources=["webcam"], type="numpy")
        with gr.Column():
            output_img = gr.Image(streaming=True)

        dep = input_img.stream(transform_cv2, [input_img, transform], [output_img],
                                time_limit=30, stream_every=0.1, concurrency_limit=30)

demo.launch(share=True)