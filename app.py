import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.predict import TailDetector

CONFIDENCE = 0.35
MODEL_PATH = Path("models/tail_model_20260226_0722.pt")

detector = TailDetector(MODEL_PATH, confidence=CONFIDENCE)


def process_image(image, confidence):
    """Обрабатывает изображение"""
    if image is None:
        return None

    print(f"\nНовое изображение, порог: {confidence}")
    detector.confidence = confidence
    result = detector.predict_with_mask(image)
    return result


#создаем интерфейс
with gr.Blocks(title="Детектор хвостов") as demo:
    gr.Markdown("#Детектор хвостов")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Загрузите фото", type="numpy")
            confidence_slider = gr.Slider(
                minimum=0.05, maximum=0.8, value=CONFIDENCE, step=0.05,
                label="Порог уверенности"
            )
            submit_btn = gr.Button("Найти хвост", variant="primary")

        with gr.Column():
            image_output = gr.Image(label="Результат")

    submit_btn.click(
        fn=process_image,
        inputs=[image_input, confidence_slider],
        outputs=image_output
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)