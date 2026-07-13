import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Forțează utilizarea procesorului (CPU) în loc de GPU
import cv2
import sys
from sign_detector import TrafficSignClassifier, detect_and_classify, draw_detections

import sys
# Adăugăm calea unde am găsit folderul pyzed direct în memoria Python
sys.path.append("/home/wsadmin/.local/lib/python3.12/site-packages")

import pyzed.sl as sl
GTSRB_LABELS_RO = {

    0: "Limita 20 km/h", 1: "Limita 30 km/h", 2: "Limita 50 km/h", 3: "Limita 60 km/h",

    4: "Limita 70 km/h", 5: "Limita 80 km/h", 6: "Sfarsit limita 80", 7: "Limita 100 km/h",

    8: "Limita 120 km/h", 9: "Depasirea interzisa", 10: "Depasire interzisa (Camioane)",

    11: "Intersectie drum fara prioritate", 12: "Drum cu prioritate", 13: "Cedeaza trecerea",

    14: "STOP", 15: "Acces interzis", 16: "Interzis autovehicule", 17: "Interzis camioane",

    18: "Alte pericole (Atentie)", 19: "Curba stanga", 20: "Curba dreapta", 21: "Curba dubla",

    22: "Drum denivelat", 23: "Drum alunecos", 24: "Ingustare drum dreapta", 25: "Lucrari",

    26: "Semafoare", 27: "Pietoni", 28: "Copii", 29: "Biciclisti", 30: "Zapada/Gheata",

    31: "Animale salbatice", 32: "Sfarsit toate restrictiile", 33: "La dreapta",

    34: "La stanga", 35: "Inainte", 36: "Inainte sau la dreapta", 37: "Inainte sau la stanga",

    38: "Ocolire prin dreapta", 39: "Ocolire prin stanga", 40: "Sens giratoriu",

    41: "Sfarsit depasire interzisa", 42: "Sfarsit depasire interzisa (Camioane)"

}


SHAPE_TO_LIKELY_CLASSES = {

    "octagon_red": {14}, # STOP

    "triangle_red": {11, 13, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}, # Avertizare / Cedeaza

    "circle_red": {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 17}, # Interzicere

    "circle_blue": {33, 34, 35, 36, 37, 38, 39, 40}, # Obligare

    "diamond_yellow": {12} # Drum cu prioritate

}import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Forțează utilizarea procesorului (CPU) în loc de GPU

import cv2

import sys

from sign_detector import TrafficSignClassifier, detect_and_classify, draw_detections


import sys

# Adăugăm calea unde am găsit folderul pyzed direct în memoria Python

sys.path.append("/home/wsadmin/.local/lib/python3.12/site-packages")


import pyzed.sl as sl


MODEL_PATH = "model_semne_100_epoci.h5"


if not os.path.exists(MODEL_PATH):

    print(f"Eroare: Nu găsesc modelul {MODEL_PATH} în acest folder!")

    sys.exit()


print("Se încarcă creierul AI... Durează câteva secunde.")

classifier = TrafficSignClassifier(MODEL_PATH)


print("Pornim camera ZED 2...")

# Îi dăm calea exactă ca text (cu ghilimele), nu ca număr

cap = cv2.VideoCapture(4)


# Setările obligatorii pentru senzorul ZED 2 (Ochi stâng + Ochi drept)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap.set(cv2.CAP_PROP_FPS, 30) # Forțăm 30 FPS pentru a evita respingerea de către Linux


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:

        print("Eroare: Nu primesc imagine de la cameră! (Dacă persistă, schimbă '/dev/video2' cu '/dev/video0' în cod)")

        break


    # Tăiem imaginea pe jumătate (ZED are 2 ochi, noi folosim doar ochiul stâng)

    inaltime, latime = frame.shape[:2]

    cadru_stang = frame[:, :latime//2]


    # Detectăm și desenăm chenarele

    detectii = detect_and_classify(cadru_stang, classifier)

    cadru_final = draw_detections(cadru_stang, detectii)


    # Afișăm rezultatul

    cv2.imshow("ZED 2 - Detectie Semne Circulatie", cadru_final)


    # Apasă tasta 'q' pentru a opri camera

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break


cap.release()

cv2.destroyAllWindows()"""

mobilenetv2_from_h5.py

-----------------------

Reconstruieste arhitectura MobileNetV2 (varianta TF-Slim/TF-Hub, alpha=0.5,

folosita de layer-ul "keras_layer_3/4" din model_semne_100_epoci.h5) DIRECT

din greutatile salvate in fisierul .h5 - fara sa mai fie nevoie sa se descarce

modelul de pe TF-Hub / Kaggle (util cand Jetson-ul / masina de build nu are

acces la internet catre kaggle.com sau storage.googleapis.com).


Toate greutatile (inclusiv backbone-ul MobileNetV2 inghetat) sunt deja in

fisierul .h5 caruia i-au fost salvate cand ati facut model.save('...h5') in

Colab, asa ca aici doar redam arhitectura standard si le incarcam 1:1 dupa

nume, direct din grupurile HDF5.

"""

import h5py

import numpy as np

import tensorflow as tf

from tensorflow.keras import layers



def _g(h5file, path):

    return np.array(h5file[path])



def _bn(x, h5file, prefix, name):

    gamma = _g(h5file, f"{prefix}/BatchNorm/gamma:0")

    beta = _g(h5file, f"{prefix}/BatchNorm/beta:0")

    mean = _g(h5file, f"{prefix}/BatchNorm/moving_mean:0")

    var = _g(h5file, f"{prefix}/BatchNorm/moving_variance:0")

    bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name)

    x = bn(x)

    bn.set_weights([gamma, beta, mean, var])

    return x



def _relu6(x):

    return layers.ReLU(6.0)(x)



def _block(x, h5file, block_name, stride, out_channels_hint, block_idx):

    in_channels = x.shape[-1]

    has_expand = f"MobilenetV2/{block_name}/expand/weights:0" in h5file


    inp = x

    if has_expand:

        w = _g(h5file, f"MobilenetV2/{block_name}/expand/weights:0")

        conv = layers.Conv2D(w.shape[-1], 1, padding="same", use_bias=False,

                              name=f"blk{block_idx}_expand")

        x = conv(x)

        conv.set_weights([w])

        x = _bn(x, h5file, f"MobilenetV2/{block_name}/expand", f"blk{block_idx}_expand_bn")

        x = _relu6(x)


    dw = _g(h5file, f"MobilenetV2/{block_name}/depthwise/depthwise_weights:0")

    pad = "same"

    if stride == 2:

        x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name=f"blk{block_idx}_pad")(x)

        pad = "valid"

    dwconv = layers.DepthwiseConv2D(3, strides=stride, padding=pad, use_bias=False,

                                     name=f"blk{block_idx}_dw")

    x = dwconv(x)

    dwconv.set_weights([dw])

    x = _bn(x, h5file, f"MobilenetV2/{block_name}/depthwise", f"blk{block_idx}_dw_bn")

    x = _relu6(x)


    pw = _g(h5file, f"MobilenetV2/{block_name}/project/weights:0")

    conv = layers.Conv2D(pw.shape[-1], 1, padding="same", use_bias=False,

                          name=f"blk{block_idx}_project")

    x = conv(x)

    conv.set_weights([pw])

    x = _bn(x, h5file, f"MobilenetV2/{block_name}/project", f"blk{block_idx}_project_bn")


    if stride == 1 and in_channels == x.shape[-1]:

        x = layers.Add(name=f"blk{block_idx}_add")([inp, x])

    return x



# stride pentru fiecare din cele 17 blocuri (standard MobileNetV2)

_STRIDES = [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]



def build_mobilenetv2_backbone(h5file, input_tensor):

    w0 = _g(h5file, "MobilenetV2/Conv/weights:0")

    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name="stem_pad")(input_tensor)

    conv0 = layers.Conv2D(w0.shape[-1], 3, strides=2, padding="valid", use_bias=False, name="stem_conv")

    x = conv0(x)

    conv0.set_weights([w0])

    x = _bn(x, h5file, "MobilenetV2/Conv", "stem_bn")

    x = _relu6(x)


    for i, stride in enumerate(_STRIDES):

        block_name = "expanded_conv" if i == 0 else f"expanded_conv_{i}"

        x = _block(x, h5file, block_name, stride, None, i)


    w1 = _g(h5file, "MobilenetV2/Conv_1/weights:0")

    conv1 = layers.Conv2D(w1.shape[-1], 1, padding="same", use_bias=False, name="head_conv")

    x = conv1(x)

    conv1.set_weights([w1])

    x = _bn(x, h5file, "MobilenetV2/Conv_1", "head_bn")

    x = _relu6(x)


    x = layers.GlobalAveragePooling2D(name="head_gap", keepdims=True)(x)


    wl = _g(h5file, "MobilenetV2/Logits/Conv2d_1c_1x1/weights:0")

    bl = _g(h5file, "MobilenetV2/Logits/Conv2d_1c_1x1/biases:0")

    logits_conv = layers.Conv2D(wl.shape[-1], 1, padding="same", use_bias=True, name="logits_conv")

    x = logits_conv(x)

    logits_conv.set_weights([wl, bl])

    x = layers.Reshape((wl.shape[-1],), name="logits_flatten")(x)

    return x  # (batch, 1001) - logits (fara softmax)



def build_full_model(h5_path, apply_softmax_on_backbone=False):

    """Reconstruieste Resizing(224) -> Rescaling(1/255) -> MobileNetV2(alpha=0.5)

    -> [softmax optional] -> Dense(43, softmax), incarcand toate greutatile

    din fisierul .h5 original (fara acces la internet)."""

    with h5py.File(h5_path, "r") as h5file:

        inp = layers.Input(shape=(32, 32, 3), name="input")

        x = layers.Resizing(224, 224, name="resizing")(inp)

        x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

        feats = build_mobilenetv2_backbone(h5file, x)

        if apply_softmax_on_backbone:

            feats = layers.Softmax(name="backbone_softmax")(feats)


        dk = _g(h5file, "dense_3/dense_3/kernel:0") if "dense_3/dense_3/kernel:0" in h5file else _g(h5file, "dense_4/dense_4/kernel:0")

        db_path = "dense_3/dense_3/bias:0" if "dense_3/dense_3/bias:0" in h5file else "dense_4/dense_4/bias:0"

        db = _g(h5file, db_path)

        dense = layers.Dense(dk.shape[-1], activation="softmax", name="final_dense")

        out = dense(feats)

        dense.set_weights([dk, db])


    model = tf.keras.Model(inp, out)

    return modelimport os

# Aceasta linie forteaza TensorFlow sa foloseasca versiunea corecta de Keras

os.environ["TF_USE_LEGACY_KERAS"] = "1"


import cv2

import numpy as np

import tensorflow as tf

import tensorflow_hub as hub

from gtsrb_labels import GTSRB_LABELS_RO, SHAPE_TO_LIKELY_CLASSES


# --- SETARI PARAMETRI ---

MIN_AREA = 500          # aria minima a unui semn (px)

MAX_AREA_RATIO = 0.5    # un semn nu poate ocupa mai mult de 50% din imagine

CONF_THRESHOLD = 0.90   # pragul de incredere

MODEL_INPUT_SIZE = 32   # dimensiunea ceruta de model


def _color_masks(hsv):

    """Returneaza mastile pentru rosu, albastru si galben (spatiul HSV)."""

    red1 = cv2.inRange(hsv, (0, 90, 60), (10, 255, 255))

    red2 = cv2.inRange(hsv, (165, 90, 60), (180, 255, 255))

    red = cv2.bitwise_or(red1, red2)


    blue = cv2.inRange(hsv, (95, 90, 50), (135, 255, 255))

    yellow = cv2.inRange(hsv, (15, 90, 90), (35, 255, 255))


    kernel = np.ones((5, 5), np.uint8)

    out = {}

    for name, m in (("red", red), ("blue", blue), ("yellow", yellow)):

        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)

        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)

        out[name] = m

    return out


def _classify_shape(cnt):

    """Numara colturile si clasifica forma matematica."""

    peri = cv2.arcLength(cnt, True)

    if peri == 0:

        return None

    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    area = cv2.contourArea(cnt)

    if area <= 0:

        return None

        

    circularity = 4 * np.pi * area / (peri * peri)

    n = len(approx)


    if n == 3:

        return "triangle"

    if n == 4:

        return "diamond"

    if 6 <= n <= 9 and circularity < 0.85:

        return "octagon"

    if circularity >= 0.72:

        return "circle"

    return None


def find_sign_candidates(frame_bgr):

    """Gaseste toate zonele din imagine care par a fi semne de circulatie."""

    h_img, w_img = frame_bgr.shape[:2]

    max_area = h_img * w_img * MAX_AREA_RATIO


    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    masks = _color_masks(hsv)


    candidates = []

    for color_name, mask in masks.items():

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < MIN_AREA or area > max_area:

                continue

                

            x, y, w, h = cv2.boundingRect(cnt)

            aspect = w / float(h)

            if aspect < 0.6 or aspect > 1.6:

                continue 

                

            shape = _classify_shape(cnt)

            if shape is None:

                continue

                

            candidates.append({

                "bbox": (x, y, w, h),

                "shape": shape,

                "color": color_name,

                "area": area,

            })


    candidates.sort(key=lambda c: -c["area"])

    kept = []

    for c in candidates:

        x, y, w, h = c["bbox"]

        overlap = False

        for k in kept:

            kx, ky, kw, kh = k["bbox"]

            ix = max(0, min(x + w, kx + kw) - max(x, kx))

            iy = max(0, min(y + h, ky + kh) - max(y, ky))

            inter = ix * iy

            if inter > 0.5 * min(w * h, kw * kh):

                overlap = True

                break

        if not overlap:

            kept.append(c)

    return kept


def shape_color_to_category(shape, color):

    if shape == "octagon" and color == "red":

        return "octagon_red"

    if shape == "triangle" and color == "red":

        return "triangle_red"

    if shape == "circle" and color == "red":

        return "circle_red"

    if shape == "circle" and color == "blue":

        return "circle_blue"

    if shape == "diamond" and color == "yellow":

        return "diamond_yellow"

    return None


class TrafficSignClassifier:

    """Clasa care foloseste modelul AI pentru a ghici semnul decupat."""

    def __init__(self, model_path):

        self.model = tf.keras.models.load_model(

            model_path,

            custom_objects={"KerasLayer": hub.KerasLayer},

            compile=False,

        )


    def predict(self, roi_bgr):

        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

        roi_rgb = cv2.resize(roi_rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))

        

        batch = np.expand_dims(roi_rgb.astype(np.float32), axis=0)

        preds = self.model.predict(batch, verbose=0)[0]

        

        cls_id = int(np.argmax(preds))

        conf = float(preds[cls_id])

        return cls_id, conf


def detect_and_classify(frame_bgr, classifier, conf_threshold=CONF_THRESHOLD, use_shape_prior=True):

    """Pipeline-ul complet: cauta, taie, clasifica, filtreaza."""

    h_img, w_img = frame_bgr.shape[:2]

    results = []

    

    for cand in find_sign_candidates(frame_bgr):

        x, y, w, h = cand["bbox"]

        

        pad = int(0.08 * max(w, h))

        x0, y0 = max(0, x - pad), max(0, y - pad)

        x1, y1 = min(w_img, x + w + pad), min(h_img, y + h + pad)

        

        roi = frame_bgr[y0:y1, x0:x1]

        if roi.size == 0:

            continue


        cls_id, conf = classifier.predict(roi)


        if use_shape_prior:

            category = shape_color_to_category(cand["shape"], cand["color"])

            if category is not None and cls_id not in SHAPE_TO_LIKELY_CLASSES.get(category, set()):

                conf *= 0.5


        if conf < conf_threshold:

            continue


        results.append({

            "bbox": (x, y, w, h),

            "class_id": cls_id,

            "label_ro": GTSRB_LABELS_RO.get(cls_id, f"clasa {cls_id}"),

            "confidence": conf,

            "shape": cand["shape"],

            "color": cand["color"],

        })


    results.sort(key=lambda r: -(r["bbox"][2] * r["bbox"][3]))

    for i, r in enumerate(results):

        r["position"] = i

    return results


def draw_detections(frame_bgr, detections):

    """Deseneaza patratele pe imaginea finala."""

    out = frame_bgr.copy()

    for det in detections:

        x, y, w, h = det["bbox"]

        

        if det["color"] == "red":

            color = (0, 0, 255)

        elif det["color"] == "blue":

            color = (255, 140, 0)

        else:

            color = (0, 200, 255)

            

        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)

        

        text = f"{det['label_ro']} ({det['confidence']*100:.0f}%)"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        cv2.rectangle(out, (x, max(0, y - th - 10)), (x + tw + 6, y), color, -1)

        cv2.putText(out, text, (x + 3, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        

    return out
MODEL_PATH = "model_semne_100_epoci.h5"

if not os.path.exists(MODEL_PATH):
    print(f"Eroare: Nu găsesc modelul {MODEL_PATH} în acest folder!")
    sys.exit()

print("Se încarcă creierul AI... Durează câteva secunde.")
classifier = TrafficSignClassifier(MODEL_PATH)

print("Pornim camera ZED 2...")
# Îi dăm calea exactă ca text (cu ghilimele), nu ca număr
cap = cv2.VideoCapture(4)

# Setările obligatorii pentru senzorul ZED 2 (Ochi stâng + Ochi drept)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30) # Forțăm 30 FPS pentru a evita respingerea de către Linux

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Eroare: Nu primesc imagine de la cameră! (Dacă persistă, schimbă '/dev/video2' cu '/dev/video0' în cod)")
        break

    # Tăiem imaginea pe jumătate (ZED are 2 ochi, noi folosim doar ochiul stâng)
    inaltime, latime = frame.shape[:2]
    cadru_stang = frame[:, :latime//2]

    # Detectăm și desenăm chenarele
    detectii = detect_and_classify(cadru_stang, classifier)
    cadru_final = draw_detections(cadru_stang, detectii)

    # Afișăm rezultatul
    cv2.imshow("ZED 2 - Detectie Semne Circulatie", cadru_final)

    # Apasă tasta 'q' pentru a opri camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()