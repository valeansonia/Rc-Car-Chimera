import numpy as np
import tensorflow as tf
import cv2
import os

# =====================================================================
# 1. ÎNCĂRCAREA MODELULUI ȘI CONFIGURAREA STRATURILOR (Grad-CAM)
# =====================================================================

# Numele fișierelor tale (asigură-te că sunt în același folder sau pune calea completă)
MODEL_PATH = "model_semne_100_epoci.h5"
VIDEO_INPUT_PATH = "WIN_20260626_14_10_38_Pro.mp4"
VIDEO_OUTPUT_PATH = "rezultat_gradcam.mp4"

print("Se încarcă modelul...")
model = tf.keras.models.load_model(MODEL_PATH)

# Pentru modelul tău bazat pe MobileNetV2, extragem substratul KerasLayer 
# și identificăm ultimul strat convoluțional din interiorul lui.
base_layer = model.get_layer("keras_layer_3")

# Construim un sub-model special pentru a calcula gradienții hărții termice
# Folosim ultimul strat convoluțional 'Conv_1/BatchNorm' din MobileNetV2
grad_model = tf.keras.models.Model(
    inputs=[base_layer.inbound_nodes[0].input_tensors],
    outputs=[base_layer.get_layer("Conv_1/BatchNorm").output, base_layer.output]
)

def compute_gradcam_heatmap(img_32x32, model, grad_model):
    """Calculează unde se uită rețeaua pentru un singur cadru."""
    # Trecem imaginea prin primele straturi de redimensionare/rescalare ale modelului tău
    x = model.get_layer("resizing_3")(img_32x32)
    x = model.get_layer("rescaling_3")(x)
    
    # Înregistrăm operațiile matematice pentru a calcula gradienții
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        # Alegem clasa cu cel mai mare scor prezis
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Calculăm gradienții clasei față de ieșirea ultimului strat convoluțional
    grads = tape.gradient(class_channel, conv_outputs)

    # Media intensității gradienților pe fiecare canal (ponderile de importanță)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Înmulțim canalele hărții finale cu ponderile calculate
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalizăm rezultatul hărții termice între 0 și 1
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

# =====================================================================
# 2. PROCESAREA CLIPULUI VIDEO CADRU CU CADRU
# =====================================================================

if not os.path.exists(VIDEO_INPUT_PATH):
    print(f"Eroare: Nu am găsit fișierul video la calea '{VIDEO_INPUT_PATH}'")
    exit()

# Deschidem videoclipul original
cap = cv2.VideoCapture(VIDEO_INPUT_PATH)

# Citim proprietățile tehnice ale clipului (lățime, înălțime, FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Configurăm obiectul care va salva noul video generat
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec standard pentru .mp4
out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

print(f"Începe procesarea video. Total cadre de procesat: {total_frames}...")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # Am ajuns la sfârșitul clipului

    # Pasul A: Pregătim cadrul curent pentru rețeaua ta neurală (modelul cere 32x32 la input)
    # OpenCV folosește formatul BGR, convertim în RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (32, 32)) 
    img_array = np.expand_dims(img_resized, axis=0) # Convertim în batch: (1, 32, 32, 3)

    # Pasul B: Generăm harta termică Grad-CAM pentru acest cadru
    heatmap = compute_gradcam_heatmap(img_array, model, grad_model)

    # Pasul C: Redimensionăm harta termică înapoi la dimensiunea originală a videoclipului tău
    heatmap_resized = cv2.resize(heatmap, (frame_width, frame_height))

    # Pasul D: Transformăm harta într-o imagine color (Paleta JET: roșu=cald/atenție maximă, albastru=rece/ignorat)
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

    # Pasul E: Suprapunem harta termică peste cadrul original al filmării (alpha controlează transparența)
    alpha = 0.4
    output_frame = cv2.addWeighted(heatmap_colored, alpha, frame, 1.0 - alpha, 0)

    # Salvăm cadrul modificat în noul fișier video
    out.write(output_frame)

    frame_count += 1
    if frame_count % 30 == 0 or frame_count == total_frames:
        print(f"Progres: {frame_count}/{total_frames} cadre procesate.")

# Eliberăm resursele din memorie
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Gata! Videoclipul cu hărțile de atenție a fost salvat cu succes sub numele: '{VIDEO_OUTPUT_PATH}'")