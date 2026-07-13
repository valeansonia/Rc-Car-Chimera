import os

import cv2
import numpy as np
import tensorflow as tf

# Ascundem GPU-ul DOAR pt TensorFlow (nu la nivel de proces/CUDA_VISIBLE_DEVICES,
# cum se facea inainte in sistem_hibrid.py) -- placa video e nevoie sa ramana
# vizibila pt SDK-ul ZED 2, care foloseste CUDA pt calculul hartii de adancime
# (distanta pana la semn). Daca ascundem GPU-ul la nivel de proces, ZED-ul nu
# mai poate calcula deloc adancimea, desi TensorFlow tot ar merge pe CPU.
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

from gtsrb_labels import (
    GTSRB_LABELS_RO,
    SHAPE_TO_LIKELY_CLASSES,
    PLAUSIBLE_SPEED_VALUES,
    GTSRB_CLASS_TO_SPEED,
    speed_confusion_group,
)


# --- SETARI PARAMETRI ---
MIN_AREA = 500
MAX_AREA_RATIO = 0.5
CONF_THRESHOLD = 0.90
MODEL_INPUT_SIZE = 64   # la fel ca modelul nou antrenat in Colab/Kaggle


def _color_masks(hsv):
    """Returneaza mastile pentru rosu, albastru si galben (spatiul HSV).

    Pragurile de luminozitate (V) minima au fost coborate mult fata de
    versiunea initiala -- verificat direct pe un semn real dintr-un video
    de test, fotografiat la lumina slaba de interior: nuanta (166-172) si
    saturatia (229-255) erau perfect corecte pt rosu, dar luminozitatea
    (V) a pixelilor rosii avea MEDIANA doar ~30 din 255 (percentila 90 =
    doar 35!) -- mult sub pragul vechi de 60, care lasa netestat aproape
    tot semnul real. Cu V_min=20, acoperirea masurata pe acel semn a urcat
    de la 0% la ~30%. La lumina de zi/exterior (V mult mai mare oricum)
    pragurile astea tot prind rosu/albastru/galben fara probleme, deci nu
    pierdem nimic din ce mergea deja -- doar acceptam si cazul intunecat.
    """
    red1 = cv2.inRange(hsv, (0, 90, 20), (10, 255, 255))
    red2 = cv2.inRange(hsv, (165, 90, 20), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    blue = cv2.inRange(hsv, (95, 90, 20), (135, 255, 255))
    yellow = cv2.inRange(hsv, (15, 90, 40), (35, 255, 255))

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
    # ATENTIE: un octogon REGULAT (ca la STOP) are circularitate ~0.948
    # (4*pi*Aria/Perimetru^2 pt un octogon ideal) -- pragul vechi de 0.85
    # era sub aceasta valoare, deci un STOP real, cu exact 8 colturi
    # corect detectate de approxPolyDP, NU trecea niciodata testul de mai
    # jos si era clasificat gresit drept "circle". Verificat direct cu un
    # octogon regulat randat sintetic. Pragul nou (0.97) lasa loc pt
    # octogoane reale (~0.90-0.95), dar tot exclude cercurile aproape
    # perfecte (circularitate foarte aproape de 1.0).
    if 6 <= n <= 9 and circularity < 0.97:
        return "octagon"
    # coborat de la 0.72 -- un semn rotund vazut usor oblic/lateral (nu
    # perfect din fata) apare ca o elipsa in imagine, cu circularitate mai
    # mica decat un cerc perfect. Prag prea strans respingea semne reale
    # vazute din alt unghi decat perfect frontal, nu doar de la distanta.
    if circularity >= 0.60:
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


def _enhance_roi_for_classification(roi_bgr, target_size=MODEL_INPUT_SIZE):
    """Pregateste crop-ul pentru clasificator, optimizat pentru semne MICI /
    de la distanta (crop-uri mici, neclare).

    Ce se schimba fata de simplul cv2.resize(...) de dinainte:
      1. Upscale cu interpolare CUBICA (nu liniara) -- pastreaza mai bine
         muchiile cifrelor/simbolurilor cand marim o imagine mica.
      2. CLAHE (contrast local adaptiv) pe canalul de luminanta -- ajuta
         mult la semnele vazute de la distanta, unde contrastul e slab.
      3. O usoara accentuare (unsharp mask) care scoate mai clar conturul
         cifrelor la limitele de viteza (30/50/60/80/100/120), principala
         cauza de confuzie intre ele.

    Nu inventeaza informatie care nu exista in imagine, dar scoate maximum
    posibil din pixelii pe care ii avem -- utila mai ales cand semnul e
    mic in cadru (departe de camera).
    """
    if roi_bgr.size == 0:
        return roi_bgr

    # 1) upscale cu interpolare cubica direct la dimensiunea modelului
    resized = cv2.resize(roi_bgr, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    # 2) CLAHE pe canalul L din LAB, ca sa nu distorsionam culorile (folosite
    #    si de shape/color prior in alta parte din pipeline)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # 3) unsharp mask usor
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    return sharpened


class TrafficSignClassifier:
    """Clasa care foloseste modelul AI (CNN custom, antrenat de la zero pe
    GTSRB) pentru a ghici semnul decupat."""

    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def predict(self, roi_bgr):
        enhanced = _enhance_roi_for_classification(roi_bgr, MODEL_INPUT_SIZE)
        roi_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

        batch = np.expand_dims(roi_rgb.astype(np.float32), axis=0)
        preds = self.model.predict(batch, verbose=0)[0]

        cls_id = int(np.argmax(preds))
        conf = float(preds[cls_id])
        return cls_id, conf

    def predict_topk(self, roi_bgr, k=3):
        """La fel ca predict(), dar returneaza si urmatoarele k-1 variante
        candidate cu scorul lor. Util pt. tracker, ca sa poata compara
        a doua cea mai probabila clasa cand semnul e neclar (de departe)."""
        enhanced = _enhance_roi_for_classification(roi_bgr, MODEL_INPUT_SIZE)
        roi_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

        batch = np.expand_dims(roi_rgb.astype(np.float32), axis=0)
        preds = self.model.predict(batch, verbose=0)[0]

        top_idx = np.argsort(preds)[::-1][:k]
        return [(int(i), float(preds[i])) for i in top_idx]


# =====================================================================
# CITIRE DIRECTA A CIFRELOR PT SEMNELE DE LIMITARE DE VITEZA
# =====================================================================
# PROBLEMA: modelul CNN (GTSRB) a fost antrenat sa recunoasca DOAR 8
# valori de viteza: 20/30/50/60/70/80/100/120. Limitele folosite frecvent
# la noi -- 90 (extravilan), 110, 130 (autostrada) -- NU EXISTA ca si
# clase in acest model. Cand camera vede un semn real de 90 sau 130,
# CNN-ul e OBLIGAT sa aleaga cea mai apropiata clasa pe care o cunoaste
# (de-aia apare mereu 90 -> 80 si 130 -> 120, nu e intamplator).
#
# SOLUTIE (fara reantrenare): citim direct cifrele din imagine, prin
# segmentare de contur + template matching cu cifre 0-9 randate cu
# fontul din OpenCV. E o metoda simpla, dar semnele de circulatie au un
# font foarte standard (negru pe fond alb/deschis, in interiorul cercului
# rosu), asa ca merge surprinzator de bine cand semnul e suficient de
# mare/clar in cadru. Daca segmentarea nu iese curat (semn prea mic/
# neclar/blurat), functia intoarce (None, 0.0) si sistemul foloseste in
# continuare rezultatul CNN-ului ca inainte -- deci nu poate strica nimic
# ce mergea deja, doar poate corecta cazurile 90/110/130.

_DIGIT_TEMPLATES = None

# Randam cifrele cu MAI MULTE fonturi OpenCV, nu doar unul -- fontul de pe
# semnele reale (DIN 1451 / similar) nu seamana perfect cu niciun font
# Hershey din OpenCV, iar cifre cu curbe mai complexe (in special "3", dar
# si "5"/"2") ies cu scor de potrivire mult mai mic pe un singur font decat
# pe altul. Comparam fiecare cifra decupata cu TOATE variantele si pastram
# scorul cel mai bun -- creste sansa sa recunoastem corect exact cifrele
# care blocau citirea unor numere intregi (ex. "3" din 30 sau 130).
_DIGIT_TEMPLATE_FONTS = (
    (cv2.FONT_HERSHEY_SIMPLEX, 2.2, 6),
    (cv2.FONT_HERSHEY_DUPLEX, 2.0, 5),
    (cv2.FONT_HERSHEY_COMPLEX, 2.0, 5),
    (cv2.FONT_HERSHEY_TRIPLEX, 2.0, 4),
)


def _build_digit_templates(size=(20, 30)):
    """Randeaza cifrele 0-9 cu MAI MULTE fonturi OpenCV, o singura data, si
    le tine in cache pt matchTemplate."""
    global _DIGIT_TEMPLATES
    if _DIGIT_TEMPLATES is not None:
        return _DIGIT_TEMPLATES

    templates = {d: [] for d in range(10)}
    for font, scale, thickness in _DIGIT_TEMPLATE_FONTS:
        for d in range(10):
            canvas = np.zeros((80, 60), dtype=np.uint8)
            cv2.putText(canvas, str(d), (5, 65), font, scale, 255, thickness, cv2.LINE_AA)
            # decupam STRANS pe cifra desenata, ca sa se alinieze cu cifrele
            # reale (care vin deja decupate strans din contur) -- altfel
            # matchTemplate compara forme cu proportii/padding diferite si
            # scorurile ies artificial de mici
            ys, xs = np.where(canvas > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            canvas = canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            canvas = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)
            templates[d].append(canvas)

    _DIGIT_TEMPLATES = templates
    return templates


def _match_digit(digit_img, size=(20, 30)):
    """Compara o cifra decupata (alb pe negru, dupa threshold) cu fiecare
    template 0-9 (toate variantele de font) si returneaza
    (cifra_cea_mai_probabila, scor)."""
    if digit_img.size == 0:
        return None, 0.0

    templates = _build_digit_templates(size)
    digit_resized = cv2.resize(digit_img, size, interpolation=cv2.INTER_AREA)

    best_d, best_score = None, -1.0
    for d, tmpl_list in templates.items():
        for tmpl in tmpl_list:
            res = cv2.matchTemplate(digit_resized, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score > best_score:
                best_score = score
                best_d = d
    return best_d, best_score


def _segment_and_read_digits(thresh, wh, ww, min_digit_conf):
    """Cauta 2-3 cifre in imaginea binara `thresh` (cifra = alb, fond =
    negru) si le citeste prin template matching. Returneaza (numar, incredere)
    sau (None, 0.0) daca segmentarea/citirea nu iese curat."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < (wh * ww) * 0.01:            # prea mic, probabil zgomot
            continue
        if ch < wh * 0.25 or ch > wh * 0.85:   # inaltimea cifrei relativ la semn
            continue
        aspect = cw / float(ch)
        if aspect > 1.15:                       # o cifra e ingusta/inalta, nu lata
            continue
        digit_boxes.append((x, y, cw, ch))

    if not (2 <= len(digit_boxes) <= 3):
        # limitele de viteza plauzibile au 2 sau 3 cifre; altfel probabil
        # nu am segmentat corect (semn neclar/de departe/umbre) -- renuntam
        return None, 0.0

    digit_boxes.sort(key=lambda b: b[0])  # ordine stanga -> dreapta

    digits = []
    confidences = []
    for (x, y, cw, ch) in digit_boxes:
        digit_crop = thresh[y:y + ch, x:x + cw]
        aspect = cw / float(ch)

        if aspect < 0.35:
            # cifra "1" e mult mai ingusta decat restul (0,3,4,6,8,9...),
            # care au toate un "corp" mai lat -- comparatia prin
            # matchTemplate se incurca usor pe forme atat de subtiri
            # (bara verticala aproape goala se potriveste artificial de
            # prost cu orice sablon), asa ca o recunoastem direct din
            # forma bounding-box-ului, fara matchTemplate.
            digits.append("1")
            confidences.append(0.9)
            continue

        d, score = _match_digit(digit_crop)
        if d is None or score < min_digit_conf:
            return None, 0.0
        digits.append(str(d))
        confidences.append(score)

    try:
        number = int("".join(digits))
    except ValueError:
        return None, 0.0

    if number not in PLAUSIBLE_SPEED_VALUES:
        return None, 0.0

    return number, float(np.mean(confidences))


def read_speed_limit_number(roi_bgr, min_digit_conf=0.55):
    """Incearca sa citeasca direct numarul de pe un semn de limitare de
    viteza (cerc rosu cu cifre negre), independent de cele 43 de clase
    GTSRB pe care le stie CNN-ul.

    roi_bgr trebuie sa fie crop-ul STRICT al semnului (fara padding mare
    in jur), ca sa nu prindem si alte forme din fundal.

    Incearca DOUA strategii de binarizare (OTSU simplu, apoi adaptiv pe
    imagine cu contrast marit prin CLAHE) -- pe camera reala, iluminarea
    neuniforma (soare/umbra pe jumatate din semn) face ca OTSU global sa
    esueze des la segmentarea cifrelor, ceea ce inseamna ca sistemul cade
    inapoi pe eticheta CNN-ului mult mai des decat ar trebui. A doua
    strategie e incercata DOAR daca prima nu produce un rezultat valid, deci
    nu schimba comportamentul in cazurile care oricum mergeau bine.

    Returneaza:
        (numar_km, incredere)  daca s-a citit clar un numar plauzibil
        (None, 0.0)            daca nu s-a putut citi clar (semn prea mic/
                                 neclar/blurat, sau nu era de fapt semn de
                                 viteza) -- in acest caz pastram decizia
                                 CNN-ului, nu inlocuim nimic.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None, 0.0

    h, w = roi_bgr.shape[:2]
    if h < 20 or w < 20:
        # prea mic ca sa segmentam cifre individuale de incredere
        return None, 0.0

    scale = 200.0 / max(h, w)
    work = cv2.resize(roi_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_CUBIC)
    wh, ww = work.shape[:2]

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    # ne concentram doar pe zona centrala a semnului, ca sa evitam
    # marginea rosie a cercului si eventualul fundal ramas din padding
    cx0, cy0 = int(ww * 0.15), int(wh * 0.15)
    cx1, cy1 = int(ww * 0.85), int(wh * 0.85)
    center_mask = np.zeros((wh, ww), dtype=np.uint8)
    center_mask[cy0:cy1, cx0:cx1] = 255

    # --- strategia 1: OTSU global (rapid, merge bine cand iluminarea e ---
    # --- uniforma pe tot semnul) ------------------------------------------
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_otsu = cv2.bitwise_and(thresh_otsu, center_mask)
    number, conf = _segment_and_read_digits(thresh_otsu, wh, ww, min_digit_conf)
    if number is not None:
        return number, conf

    # --- strategia 2 (fallback): CLAHE + threshold adaptiv, pt semne cu ---
    # --- iluminare neuniforma / contrast slab, unde OTSU global esueaza --
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    thresh_adapt = cv2.adaptiveThreshold(
        gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=25, C=7,
    )
    thresh_adapt = cv2.bitwise_and(thresh_adapt, center_mask)
    # curatam zgomotul de sare-si-piper introdus des de thresholding adaptiv
    kernel = np.ones((2, 2), np.uint8)
    thresh_adapt = cv2.morphologyEx(thresh_adapt, cv2.MORPH_OPEN, kernel)
    return _segment_and_read_digits(thresh_adapt, wh, ww, min_digit_conf)


def classify_speed_limit(roi_bgr, roi_bgr_strict, cnn_cls_id, cnn_conf, min_digit_conf=0.55):
    """Decide clasa finala pt un candidat de semn de limitare de viteza,
    combinand CNN-ul cu citirea directa a cifrelor (OCR).

    De ce NU dam OCR-ului prioritate necondiționata peste tot: OCR-ul poate
    citi gresit (blur, unghi, digitizare proasta), la fel ca CNN-ul. Pt
    valorile pe care CNN-ul le stie sigur (20/30/50/70 -- in afara
    grupurilor de confuzie), un citit OCR gresit ocazional ar suprascrie o
    clasificare CNN deja corecta, ceea ce inseamna ca am strica exact acele
    cazuri care mergeau bine. Asa ca ii dam OCR-ului prioritate necondiționata
    DOAR acolo unde chiar are rost:
      - valori care NU exista deloc ca clasa GTSRB (90/110/130) -- CNN e
        structural incapabil sa le spuna, deci OCR e SINGURA sursa posibila;
      - valori din grupurile de confuzie cunoscute (60/80/90 si
        100/110/120/130, vezi SPEED_CONFUSION_GROUPS din gtsrb_labels.py) --
        aici CNN e nesigur prin constructie (cifre prea asemanatoare).
    In afara acestor cazuri (20/30/50/70), folosim OCR doar daca e cel putin
    la fel de increzator ca CNN-ul -- exact ca inainte de acest fix.

    Returneaza (class_id, confidence, ocr_used).
    """
    ocr_roi = roi_bgr_strict if roi_bgr_strict is not None and roi_bgr_strict.size > 0 else roi_bgr
    ocr_speed, ocr_conf = read_speed_limit_number(ocr_roi, min_digit_conf=min_digit_conf)

    if ocr_speed is None:
        return cnn_cls_id, cnn_conf, False

    if ocr_speed not in GTSRB_CLASS_TO_SPEED.values():
        # valoare care NU exista ca si clasa GTSRB (90/110/130 etc.) -- CNN
        # nu poate spune NICIODATA acest numar, deci OCR e singura sursa
        # posibila -- folosim un "id virtual" negativ (vezi label_for_class
        # din gtsrb_labels.py)
        return -ocr_speed, ocr_conf, True

    matched_class = next(k for k, v in GTSRB_CLASS_TO_SPEED.items() if v == ocr_speed)
    cnn_speed = GTSRB_CLASS_TO_SPEED.get(cnn_cls_id)
    ocr_group = speed_confusion_group(ocr_speed)
    cnn_group = speed_confusion_group(cnn_speed) if cnn_speed is not None else None
    # ATENTIE: trebuie sa fie ACELASI grup pe ambele parti, nu doar "unul
    # din cele doua e membru al vreunui grup" -- altfel un CNN corect (ex.
    # 30, care nu e in niciun grup) ar fi suprascris de un OCR care citeste
    # gresit o valoare care intamplator e in grup (ex. 80), desi cele doua
    # citiri n-au nicio legatura reala intre ele.
    same_confusion_pair = ocr_group is not None and ocr_group == cnn_group

    if same_confusion_pair or ocr_conf >= cnn_conf:
        return matched_class, max(cnn_conf, ocr_conf), True

    # in afara zonei de confuzie si OCR mai putin increzator decat CNN-ul --
    # pastram decizia CNN-ului (deja fiabila aici), ca sa nu o stricam cu un
    # citit OCR ocazional gresit
    return cnn_cls_id, cnn_conf, False


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