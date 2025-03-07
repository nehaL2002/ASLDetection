import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import gradio as gr
from PIL import Image

model = YOLO('aslmodel.pt')

def predict_image(img, conf_threshold, iou_threshold):
    """Predicts objects in an image using a YOLOv8 model with adjustable confidence and IOU thresholds."""
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=800,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im

iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(sources="webcam", type="pil", label="Capture Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    live=True,  # Enables real-time processing
    title="Ultralytics Gradio",
    description="Capture images from your webcam for real-time inference using the Ultralytics YOLOv8n model.",
)

if __name__ == "__main__":
    iface.launch()