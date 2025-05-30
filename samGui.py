import gradio as gr
import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
import sys  # Sunucuyu kapatmak için gerekli
import logging  # Loglama için gerekli
import torch  # Donanım bilgisi için gerekli
import os  # CUDA bellek yönetimi için gerekli
import urllib.request  # Dosya indirme için gerekli
import cv2  # Görüntü işleme için gerekli

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("application.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# CUDA bellek yönetimi ayarı
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# GPU belleğini kontrol et ve yetersizse CPU'ya geç
def check_and_set_device(required_memory_mb=6144):  # En az 6 GB boş bellek gereksinimi
    if torch.cuda.is_available():
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        free_memory_mb = free_memory / (1024 * 1024)  # Byte to MB
        logging.info(f"GPU belleği kontrol ediliyor: {free_memory_mb:.2f} MB boş.")
        if free_memory_mb < required_memory_mb:
            logging.warning(f"GPU belleği yetersiz ({free_memory_mb:.2f} MB). CPU'ya geçiliyor...")
            return torch.device("cpu")
        else:
            logging.info("GPU belleği yeterli. GPU kullanılacak.")
            return torch.device("cuda")
    else:
        logging.warning("GPU bulunamadı. CPU kullanılacak.")
        return torch.device("cpu")

# SAM modelini yükleme fonksiyonu
def load_sam_model(model_type):
    logging.info(f"SAM modeli yükleniyor: {model_type}")
    sam_checkpoint_map = {
        "vit_b": "sam_vit_b_01ec64.pth",
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth"
    }
    sam_checkpoint = sam_checkpoint_map[model_type]
    if not os.path.exists(sam_checkpoint):
        logging.info(f"{model_type} modeli indiriliyor...")
        url_map = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
        }
        urllib.request.urlretrieve(url_map[model_type], sam_checkpoint)
        logging.info(f"{model_type} modeli başarıyla indirildi.")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    logging.info(f"SAM modeli başarıyla yüklendi: {model_type}")
    return SamPredictor(sam)

# GPU belleğini kontrol et ve cihazı ayarla
device = check_and_set_device(required_memory_mb=6144)  # En az 6 GB boş bellek gereksinimi

# Başlangıçta varsayılan model
model_type = "vit_l"  # Varsayılan model
predictor = load_sam_model(model_type)

# Yeni bir fonksiyon: GPU belleği yetersizse işlemleri CPU'ya taşı
def safe_predict(predictor, *args, **kwargs):
    try:
        # GPU'da çalıştırmayı dene
        return predictor.predict(*args, **kwargs)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "CUDA out of memory" in str(e):
            logging.warning("GPU belleği yetersiz. İşlem CPU'ya taşınıyor...")
            predictor.model.to("cpu")  # Modeli CPU'ya taşı
            result = predictor.predict(*args, **kwargs)
            predictor.model.to(device)  # İşlemden sonra modeli tekrar GPU'ya taşı
            return result
        else:
            raise e

# Maske üretiminde gölgeleri de dahil etmek için morfolojik genişletme fonksiyonu ekleniyor
def include_shadows(mask: np.ndarray, dilation_iter=10):
    """Maske üzerinde morfolojik genişletme ile gölgeleri de dahil et."""
    kernel = np.ones((7, 7), np.uint8)
    mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=dilation_iter)
    return mask_dilated

# Yeni bir fonksiyon: Tıklama noktalarını işlemek ve maske oluşturmak için
def generate_mask_with_click(image, coordinates):
    x, y = coordinates
    logging.info(f"Kullanıcı {x}, {y} koordinatlarına tıkladı. Maske oluşturuluyor...")

    # Görüntüyü SAM modeline uygun şekilde işleme
    image = np.array(image)
    predictor.set_image(image)

    # Tıklama noktasını kullanarak maske oluşturma
    input_point = np.array([[x, y]])
    input_label = np.array([1])  # 1: foreground (ön plan)
    masks, _, _ = safe_predict(predictor, point_coords=input_point, point_labels=input_label, box=None)

    # İlk maskeyi al ve gölgeleri dahil et
    mask = masks[0]
    mask_with_shadows = include_shadows(mask, dilation_iter=10)
    mask_image = Image.fromarray((mask_with_shadows * 255).astype(np.uint8))
    logging.info("Maske (gölgeler dahil) başarıyla oluşturuldu.")
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
    return generate_mask_with_click(image, (x, y))

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
    new_mask = generate_mask_with_click(image, (x, y))

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

# Yeni bir fonksiyon: Birden fazla seçimi birleştirerek tek bir maske oluşturmak için
def combine_masks(image, coordinates_list):
    if not coordinates_list:
        logging.warning("Maske birleştirmek için önce birden fazla nokta seçin.")
        return None
    logging.info(f"{len(coordinates_list)} nokta seçildi. Maskeler birleştiriliyor...")

    # Görüntüyü SAM modeline uygun şekilde işleme
    image = np.array(image)
    predictor.set_image(image)

    combined_mask = None
    for x, y in coordinates_list:
        logging.info(f"{x}, {y} koordinatları için maske oluşturuluyor...")
        input_point = np.array([[x, y]])
        input_label = np.array([1])  # 1: foreground (ön plan)
        masks, _, _ = safe_predict(predictor, point_coords=input_point, point_labels=input_label, box=None)

        # İlk maskeyi al ve gölgeleri dahil et
        mask = masks[0]
        mask_with_shadows = include_shadows(mask, dilation_iter=10)
        if combined_mask is None:
            combined_mask = mask_with_shadows
        else:
            combined_mask = np.maximum(combined_mask, mask_with_shadows)

    # Birleştirilmiş maskeyi döndür
    combined_mask_image = Image.fromarray((combined_mask * 255).astype(np.uint8))
    logging.info("Tüm maskeler (gölgeler dahil) başarıyla birleştirildi.")
    return combined_mask_image

# Yeni bir wrapper fonksiyon: Birden fazla tıklama koordinatını saklamak için
def handle_multi_select(image, evt: gr.SelectData, coordinates_list):
    if evt is None or evt.index is None:
        logging.warning("Geçerli bir tıklama algılanmadı.")
        return None, coordinates_list
    x, y = evt.index
    logging.info(f"Kullanıcı {x}, {y} koordinatlarına tıkladı.")
    coordinates_list.append((x, y))
    return zoom_preview(image, evt), coordinates_list

# Yeni bir wrapper fonksiyon: Birden fazla tıklama koordinatını saklamak ve zoom penceresi oluşturmak için
def handle_multi_select_with_zoom(image, evt: gr.SelectData, coordinates_list, zoom_previews):
    if evt is None or evt.index is None:
        logging.warning("Geçerli bir tıklama algılanmadı.")
        return zoom_previews, coordinates_list
    x, y = evt.index
    logging.info(f"Kullanıcı {x}, {y} koordinatlarına tıkladı.")
    coordinates_list.append((x, y))
    zoomed_image = zoom_preview(image, evt)
    zoom_previews.append(zoomed_image)
    return zoom_previews, coordinates_list

# Yeni bir fonksiyon: Seçilen maskeleri silmek için
def remove_last_selection(coordinates_list, zoom_previews):
    if not coordinates_list:
        logging.warning("Silinecek seçim bulunamadı.")
        return zoom_previews, coordinates_list
    logging.info(f"Son seçim siliniyor: {coordinates_list[-1]}")
    coordinates_list.pop()
    zoom_previews.pop()
    return zoom_previews, coordinates_list

# Belirli bir noktayı silmek için fonksiyon (eksikti, ekleniyor)
def remove_specific_selection(coordinates_list, zoom_previews, index):
    if index is None or index < 0 or index >= len(coordinates_list):
        logging.warning("Geçerli bir seçim indeksi bulunamadı.")
        return zoom_previews, coordinates_list
    logging.info(f"Seçim siliniyor: {coordinates_list[index]}")
    del coordinates_list[index]
    del zoom_previews[index]
    return zoom_previews, coordinates_list

# Yeni bir fonksiyon: Maskeyi asıl resim üzerinde göstermek için
def overlay_mask_on_image(image, mask):
    if mask is None:
        logging.warning("Gösterilecek maske bulunamadı.")
        return image
    logging.info("Maske asıl resim üzerine bindiriliyor...")
    image_np = np.array(image)
    mask_np = np.array(mask)

    # Maskeyi kırmızı renkle bindir
    overlay = image_np.copy()
    overlay[mask_np > 0] = [255, 0, 0]  # Kırmızı renk
    combined_image = Image.fromarray(overlay)
    return combined_image

# Yeni bir fonksiyon: Kullanılan donanım bilgisini almak için
def get_device_info():
    if torch.cuda.is_available():
        device_info = f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}"
    else:
        device_info = "CPU: Kullanılıyor"
    logging.info(f"Kullanılan donanım: {device_info}")
    return device_info

# Maskeyi indirmek için fonksiyon
def download_mask(mask):
    if mask is None:
        logging.warning("İndirilecek maske bulunamadı.")
        return
    logging.info("Maske indiriliyor...")
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray((mask * 255).astype(np.uint8))
    mask_path = os.path.join(os.getcwd(), "mask.png")
    mask.save(mask_path)
    logging.info(f"Maske kaydedildi: {mask_path}")
    # Kullanıcıya indirme linki sunmak için dosya yolunu döndür
    return gr.File(mask_path)

try:
    # Gradio arayüzü
    logging.info("Gradio arayüzü başlatılıyor...")
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("## SAM ile Görüntü Segmentasyonu")
                gr.Markdown("""
                ### Kullanım Kılavuzu:
                1. **Resim Yükle**: Segmentasyon yapmak istediğiniz resmi yükleyin.
                2. **Tıklama**: Resim üzerinde birden fazla noktaya tıklayın. Tıkladığınız noktalar birleştirilerek tek bir maske oluşturulacaktır.
                3. **Zoom Önizleme**: Her tıklama için bir zoom penceresi oluşturulacaktır.
                4. **Maske Oluştur**: 'Maske Oluştur' düğmesine tıklayarak seçilen tüm noktalara göre birleştirilmiş bir maske oluşturabilirsiniz.
                5. **Seçimi Sil**: Listeden bir seçim yaparak belirli bir noktayı ve ilgili zoom önizlemesini kaldırabilirsiniz.
                6. **Maske Temizle**: 'Maske Temizle' düğmesine tıklayarak tüm seçimleri ve maskeleri temizleyebilirsiniz.
                7. **Maske İndir**: 'Maske İndir' düğmesine tıklayarak oluşturulan maskeyi indirebilirsiniz.
                8. **Yeni Resim**: 'Yeni Resim' düğmesine tıklayarak tüm seçimleri ve maskeleri sıfırlayabilirsiniz.
                9. **Model Seçimi**: Aşağıdaki menüden SAM modelini seçebilirsiniz.
                10. **Sonuç**: Zoom önizlemeleri ve oluşturulan maske sağ tarafta görüntülenecektir.
                """)
                device_info_output = gr.Textbox(label="Kullanılan Donanım", value=get_device_info(), interactive=False)
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    label="SAM Modeli Seçimi",
                    choices=["vit_b", "vit_h", "vit_l"],
                    value="vit_l",
                    interactive=True
                )
                mask_button = gr.Button("Maske Oluştur")
                remove_selection_button = gr.Button("Seçimi Sil")
                clear_mask_button = gr.Button("Maske Temizle")
                reset_button = gr.Button("Yeni Resim")
                download_button = gr.Button("Maske İndir")
                update_dropdown_button = gr.Button("Seçim Listesini Güncelle")
                selection_dropdown = gr.Dropdown(label="Seçimi Sil", choices=[], interactive=True)
        with gr.Row():
            image_input = gr.Image(label="Resim Yükle", type="pil", interactive=True)
            zoom_previews_output = gr.Gallery(label="Zoom Önizlemeleri", columns=3, height="400px")
            mask_output = gr.Image(label="Oluşturulan Maske")
            overlay_output = gr.Image(label="Asıl Resim Üzerinde Maske")
        coordinates_list = gr.State([])  # Tıklanan tüm koordinatları saklamak için
        zoom_previews = gr.State([])  # Tüm zoom önizlemelerini saklamak için

        # Model seçimi değiştiğinde SAM modelini yeniden yükle
        model_selector.change(
            lambda selected_model: load_sam_model(selected_model),
            inputs=[model_selector],
            outputs=[]
        )

        # Tıklama ile zoom yapma ve koordinatları saklama
        image_input.select(
            handle_multi_select_with_zoom,
            inputs=[image_input, coordinates_list, zoom_previews],
            outputs=[zoom_previews_output, coordinates_list]
        )

        # Maske oluşturma düğmesi
        mask_button.click(
            lambda image, coordinates: (combine_masks(image, coordinates), overlay_mask_on_image(image, combine_masks(image, coordinates))),
            inputs=[image_input, coordinates_list],
            outputs=[mask_output, overlay_output]
        )

        # Seçimi silme düğmesi
        remove_selection_button.click(
            remove_specific_selection,
            inputs=[coordinates_list, zoom_previews, selection_dropdown],
            outputs=[zoom_previews_output, coordinates_list]
        )

        # Maske temizleme düğmesi
        clear_mask_button.click(
            lambda: ([], [], None, None),  # Zoom önizlemelerini, koordinat listesini, maskeyi ve overlay'i temizle
            inputs=[],
            outputs=[zoom_previews_output, coordinates_list, mask_output, overlay_output]
        )

        # Yeni resim düğmesi
        reset_button.click(
            lambda: (None, [], [], None, None),  # Resmi, koordinat listesini, zoom önizlemelerini ve maskeleri sıfırla
            inputs=[],
            outputs=[image_input, coordinates_list, zoom_previews, mask_output, overlay_output]
        )

        # Maske indirme düğmesi
        download_button.click(
            download_mask,
            inputs=[mask_output],
            outputs=[gr.File()]  # Kullanıcıya indirme linki sun
        )

        # Logları göstermek için bir Textbox ekliyoruz
        with gr.Row():
            log_output = gr.Textbox(label="Loglar", value="", interactive=False, lines=10)

        # Logları güncellemek için bir fonksiyon
        def update_logs():
            try:
                with open("application.log", encoding="utf-8", errors="ignore") as log_file:
                    return "\n".join(log_file.readlines()[-10:])  # Son 10 log satırını döndür
            except FileNotFoundError:
                return "Log dosyası bulunamadı."

        # Logları düzenli olarak güncellemek için bir düğme ekliyoruz
        log_refresh_button = gr.Button("Logları Güncelle")
        log_refresh_button.click(
            update_logs,
            inputs=[],
            outputs=[log_output]
        )

        # Sunucuyu kapatma düğmesi
        stop_button = gr.Button("Sunucuyu Kapat")
        def stop_server():
            logging.info("Sunucu kullanıcı isteğiyle kapatıldı.")
            sys.exit(0)
        stop_button.click(
            stop_server,
            inputs=[],
            outputs=[]
        )

    demo.launch(share=True)
    logging.info("Gradio arayüzü başarıyla başlatıldı.")
except Exception as e:
    logging.error(f"Hata oluştu: {e}")
    sys.exit(1)  # Sunucuyu kapat