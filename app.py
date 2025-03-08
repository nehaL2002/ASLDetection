import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import gradio as gr
from PIL import Image

model = YOLO('aslmodel.pt')

def detection_image(image):
    results = model.predict(source=image, conf=0.25)
    for result in results:
        img = result.plot()  # Get the annotated image (numpy array)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        pil_img = Image.fromarray(img_rgb)  # Convert to PIL image
        return pil_img

demo = gr.Interface(
    fn=detection_image,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Image(),
    title="ASL Alphabetic Letter Detection",
    description="_Using YOLOv8_"
)

if __name__ == "__main__":
    demo.launch()
