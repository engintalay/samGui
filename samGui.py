import gradio as gr
import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
import sys  # Sunucuyu kapatmak için gerekli
import logging  # Loglama için gerekli

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("application.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# SAM modelini yükleme
logging.info("SAM modeli yükleniyor...")
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Model dosyasını indirip bu yola koymalısınız
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)
logging.info("SAM modeli başarıyla yüklendi.")

# Yeni bir fonksiyon: Tıklama noktalarını işlemek ve maske oluşturmak için
def generate_mask_with_click(image, evt: gr.SelectData):
    x, y = evt.index
    logging.info(f"Kullanıcı {x}, {y} koordinatlarına tıkladı. Maske oluşturuluyor...")

    # Görüntüyü SAM modeline uygun şekilde işleme
    image = np.array(image)
    predictor.set_image(image)

    # Tıklama noktasını kullanarak maske oluşturma
    input_point = np.array([[x, y]])
    input_label = np.array([1])  # 1: foreground (ön plan)
    masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, box=None)

    # İlk maskeyi döndürme
    mask = masks[0]
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    logging.info("Maske başarıyla oluşturuldu.")
    return mask_image

# Yeni bir fonksiyon: Zoom önizlemesi için
def zoom_preview(image, evt: gr.SelectData):
    x, y = evt.index
    logging.info(f"Kullanıcı {x}, {y} koordinatlarında zoom önizlemesi yapıyor...")

    zoom_size = 50  # Zoom boyutu
    zoom_factor = 2  # Zoom oranı

    # Görüntüyü numpy array'e çevir
    image_np = np.array(image)

    # Zoom bölgesini hesapla
    x_min = max(0, x - zoom_size)
    x_max = min(image_np.shape[1], x + zoom_size)
    y_min = max(0, y - zoom_size)
    y_max = min(image_np.shape[0], y + zoom_size)

    # Zoom bölgesini al ve büyüt
    zoom_region = image_np[y_min:y_max, x_min:x_max]
    zoomed_image = Image.fromarray(np.kron(zoom_region, np.ones((zoom_factor, zoom_factor, 1))).astype(np.uint8))
    logging.info("Zoom önizlemesi başarıyla oluşturuldu.")
    return zoomed_image

# Yeni bir fonksiyon: Tıklama noktalarını işlemek ve zoom yapmak için
def handle_zoom(image, evt: gr.SelectData):
    if evt is None or evt.index is None:
        logging.warning("Zoom yapmak için geçerli bir tıklama algılanmadı.")
        return None  # Zoom önizlemesi için boş bir değer döndür
    x, y = evt.index
    logging.info(f"Kullanıcı {x}, {y} koordinatlarında zoom yapıyor...")
    return zoom_preview(image, evt)

# Yeni bir fonksiyon: Maske oluşturma düğmesi için
def handle_mask(image, coordinates):
    if coordinates is None:
        logging.warning("Maske oluşturmak için önce bir noktaya tıklayın.")
        return None
    logging.info(f"Maske oluşturuluyor. Son tıklanan koordinatlar: {coordinates}")
    x, y = coordinates
    return generate_mask_with_click(image, gr.SelectData(index=(x, y)))

# Yeni bir fonksiyon: Maskeyi temizlemek için
def clear_mask():
    logging.info("Maskeler temizleniyor...")
    return None  # Maskeyi temizlemek için None döndür

# Yeni bir fonksiyon: Maskeyi birleştirmek için
def add_mask(image, coordinates, current_mask):
    if coordinates is None:
        logging.warning("Maske eklemek için önce bir noktaya tıklayın.")
        return current_mask, current_mask  # Mevcut maskeyi iki kez döndür

    logging.info(f"Yeni maske ekleniyor. Son tıklanan koordinatlar: {coordinates}")
    x, y = coordinates
    new_mask = generate_mask_with_click(image, gr.SelectData(index=(x, y)))

    if current_mask is None:
        return new_mask, new_mask  # Eğer mevcut maske yoksa, yeni maskeyi iki kez döndür

    # Mevcut maske ile yeni maskeyi birleştir
    combined_mask = Image.fromarray(
        np.maximum(np.array(current_mask), np.array(new_mask))
    )
    logging.info("Maskeler başarıyla birleştirildi.")
    return combined_mask, combined_mask  # Birleştirilmiş maskeyi iki kez döndür

# Yeni bir wrapper fonksiyon: Zoom ve tıklama koordinatlarını işlemek için
def handle_select(image, evt: gr.SelectData):
    if evt is None or evt.index is None:
        logging.warning("Zoom yapmak için geçerli bir tıklama algılanmadı.")
        return None, None  # Zoom önizlemesi ve koordinatlar için boş değerler döndür
    logging.info(f"Kullanıcı {evt.index[0]}, {evt.index[1]} koordinatlarına tıkladı.")
    zoomed_image = zoom_preview(image, evt)
    return zoomed_image, evt.index

try:
    # Gradio arayüzü
    logging.info("Gradio arayüzü başlatılıyor...")
    with gr.Blocks() as demo:
        gr.Markdown("## SAM ile Görüntü Segmentasyonu")
        gr.Markdown("""
        ### Kullanım Kılavuzu:
        1. **Resim Yükle**: Segmentasyon yapmak istediğiniz resmi yükleyin.
        2. **Tıklama**: Resim üzerinde bir noktaya tıklayın. Tıkladığınız noktaya göre zoom yapılacaktır.
        3. **Maske Oluştur**: 'Maske Oluştur' düğmesine tıklayarak son tıklanan noktaya göre maske oluşturabilirsiniz.
        4. **Maske Ekle**: 'Maske Ekle' düğmesine tıklayarak mevcut maskeye yeni bir maske ekleyebilirsiniz.
        5. **Maske Temizle**: 'Maske Temizle' düğmesine tıklayarak tüm maskeleri temizleyebilirsiniz.
        6. **Sonuç**: Zoom önizlemesi ve oluşturulan maske sağ tarafta görüntülenecektir.
        """)
        with gr.Row():
            image_input = gr.Image(label="Resim Yükle", type="pil", interactive=True)
            zoom_output = gr.Image(label="Zoom Önizleme")
            mask_output = gr.Image(label="Oluşturulan Maske")
        last_click = gr.State(None)  # Son tıklanan koordinatları saklamak için
        current_mask = gr.State(None)  # Mevcut maskeyi saklamak için

        # Tıklama ile zoom yapma
        image_input.select(
            handle_select,
            inputs=[image_input],
            outputs=[zoom_output, last_click]
        )

        # Maske oluşturma düğmesi
        mask_button = gr.Button("Maske Oluştur")
        mask_button.click(
            handle_mask,
            inputs=[image_input, last_click],
            outputs=[mask_output]
        )

        # Maske ekleme düğmesi
        add_mask_button = gr.Button("Maske Ekle")
        add_mask_button.click(
            add_mask,
            inputs=[image_input, last_click, current_mask],
            outputs=[mask_output, current_mask]
        )

        # Maske temizleme düğmesi
        clear_mask_button = gr.Button("Maske Temizle")
        clear_mask_button.click(
            clear_mask,
            inputs=[],
            outputs=[mask_output, current_mask]
        )

    demo.launch()
    logging.info("Gradio arayüzü başarıyla başlatıldı.")
except Exception as e:
    logging.error(f"Hata oluştu: {e}")
    sys.exit(1)  # Sunucuyu kapat