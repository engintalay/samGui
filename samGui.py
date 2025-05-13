import gradio as gr
import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

# SAM modelini yükleme
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Model dosyasını indirip bu yola koymalısınız
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

def generate_mask(image, x, y, width, height):
    # Görüntüyü SAM modeline uygun şekilde işleme
    image = np.array(image)
    predictor.set_image(image)

    # Kullanıcıdan gelen seçim (x, y, width, height) ile maske oluşturma
    input_box = np.array([x, y, x + width, y + height])
    masks, _, _ = predictor.predict(box=input_box, point_coords=None, point_labels=None)

    # İlk maskeyi döndürme
    mask = masks[0]
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    return mask_image

# Gradio arayüzü
with gr.Blocks() as demo:
    gr.Markdown("## SAM ile Görüntü Segmentasyonu")
    with gr.Row():
        image_input = gr.Image(label="Resim Yükle", type="pil")
        x_input = gr.Number(label="X Koordinatı")
        y_input = gr.Number(label="Y Koordinatı")
        width_input = gr.Number(label="Genişlik")
        height_input = gr.Number(label="Yükseklik")
    mask_output = gr.Image(label="Oluşturulan Maske")
    submit_button = gr.Button("Maske Oluştur")

    submit_button.click(
        generate_mask,
        inputs=[image_input, x_input, y_input, width_input, height_input],
        outputs=mask_output,
    )

demo.launch()