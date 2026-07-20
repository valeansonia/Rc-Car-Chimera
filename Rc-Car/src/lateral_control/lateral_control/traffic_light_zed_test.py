import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    import tensorflow_hub as hub
except ImportError:
    hub = None

try:
    import tf_keras as keras
except ImportError:
    try:
        from tensorflow import keras
    except ImportError:
        keras = None

try:
    import pyzed.sl as sl
except ImportError:
    sl = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


DEFAULT_YOLO_MODEL_PATH = (
    "/home/wsadmin/Desktop/Rc-Car-Chimera/Rc-Car/src/model/bestSem.pt"
)


LABELS = ["back", "green", "red", "yellow"]

MODEL_FILE_NAMES = [
    "traffic_light_classifier.h5",
    "mobilenetv2-Adam.h5",
]

DISPLAY_COLORS = {
    "back": (160, 160, 160),
    "green": (0, 200, 0),
    "red": (0, 0, 255),
    "yellow": (0, 220, 255),
    "unknown": (160, 160, 160),
}

ACTION_BY_LABEL = {
    "red": "STOP",
    "yellow": "SLOW",
    "green": "GO",
    "back": "NO_TRAFFIC_LIGHT",
    "unknown": "NO_TRAFFIC_LIGHT",
}

COLOR_RANGES = {
    "red": [
        # Pastram aici numai rosul propriu-zis. LED-ul galben al semaforului
        # este redat de camera ZED mai portocaliu si ajunge in zona H=5..14.
        ((0, 80, 100), (4, 255, 255)),
        ((168, 80, 110), (180, 255, 255)),
    ],
    "yellow": [
        # Calibrat din capturile ZED: LED-ul portocaliu are H predominant
        # 8..13. Pastram o marja 5..20, fara zona verde-galbuie de peste 20.
        ((5, 45, 80), (20, 255, 255)),
    ],
    "green": [
        # Zona 28..29 este ambigua: verdele indepartat este deplasat aici de
        # white balance. O suprapunem cu galbenul si o rezolvam mai jos din
        # raportul G/R si pozitia becului.
        ((28, 35, 55), (90, 255, 255)),
        ((90, 35, 70), (98, 255, 255)),
    ],
}

COLOR_OVERRIDE_MIN_CONFIDENCE = 0.18
MIN_TRAFFIC_LIGHT_HEIGHT_WIDTH_RATIO = 1.35
MIN_TRAFFIC_LIGHT_BBOX_HEIGHT = 14
MIN_NEW_TRAFFIC_LIGHT_BBOX_HEIGHT = 20
DETECTION_HORIZONTAL_MARGIN_RATIO = 0.15
COLOR_CONFIRM_FRAMES = 3
RED_CONFIRM_FRAMES = 2
# Galbenul inseamna SLOW si trebuie acceptat imediat; altfel un LED ambre
# detectat intermitent nu ajunge niciodata la doua cadre consecutive.
YELLOW_CONFIRM_FRAMES = 1
STABLE_COLOR_LABEL = None
PENDING_COLOR_LABEL = None
PENDING_COLOR_COUNT = 0
# Permitem masurarea apropiata. Corectia speciala este folosita numai cand
# bbox-ul semaforului este mare; distantele medii/departate raman neschimbate.
ZED_DEPTH_MIN_M = 0.10
ZED_DEPTH_MAX_M = 20.0
# Folosim numai centrul de 40% al bbox-ului pentru ZED. Valoarea veche de
# 0.30 evita marginile carcasei, unde intra fundalul si distanta sarea.
DISTANCE_BOX_MARGIN_RATIO = 0.30
# Aceleasi valori cu care distanta verdelui a raspuns corect in test:
# mediana pe 5 cadre elimina citirile izolate, iar alpha=0.15 urmareste
# apropierea fara intarzierea mare produsa de fereastra de 7 cadre.
DISTANCE_HISTORY_LEN = 5
DISTANCE_SMOOTHING_ALPHA = 0.15
DISTANCE_HISTORY = deque(maxlen=DISTANCE_HISTORY_LEN)
SMOOTHED_DISTANCE_M = None
DISTANCE_TRACK_BBOX = None
CLOSE_DISTANCE_BBOX_HEIGHT = 100
CLOSE_DISTANCE_CONFIRM_M = 0.35
CLOSE_DISTANCE_CONFIRM_FRAMES = 3
# Sub aproximativ 30 cm, stereo-depth-ul ZED poate ramane blocat pe fundal,
# chiar daca depth_minimum_distance este setat mai jos. Retinem ultima
# masurare ZED buna cat timp carcasa este inca suficient de mica si, numai
# cand bbox-ul creste clar, extrapolam distanta prin perspectiva (d ~ 1/h).
CLOSE_DISTANCE_ANCHOR_MAX_BBOX_HEIGHT = 80
CLOSE_DISTANCE_SWITCH_BBOX_HEIGHT = 90
CLOSE_DISTANCE_ZED_TOLERANCE_M = 0.08
CLOSE_DISTANCE_ANCHOR_M = None
CLOSE_DISTANCE_ANCHOR_BBOX_HEIGHT = None
CLOSE_DISTANCE_MODE_ACTIVE = False
LAST_RELIABLE_YOLO_BBOX = None
LAST_RELIABLE_YOLO_LABEL = None
MAX_MISSED_DETECTION_FRAMES = 20

# Fallback pentru LED-ul galben/portocaliu. YOLO-ul poate rata complet
# semaforul ambre; in acel caz HSV trebuie sa poata propune singur un bbox.
# H=0..3 este de regula LED-ul rosu foarte intens. Fallback-ul acesta trebuie
# sa caute numai ambre; rosul ramane detectat de YOLO + analiza din bbox.
AMBER_FALLBACK_H_MIN = 5
AMBER_FALLBACK_H_MAX = 20
AMBER_FALLBACK_S_MIN = 55
AMBER_FALLBACK_V_MIN = 140
AMBER_FALLBACK_MIN_AREA = 4
AMBER_FALLBACK_MAX_AREA_RATIO = 0.01
AMBER_FALLBACK_MIN_SCORE = 0.15
AMBER_FALLBACK_HORIZONTAL_MARGIN_RATIO = 0.30
# In test, semaforul este in jumatatea superioara. Sub aceasta limita apar
# LED-urile placilor de pe masa, care provocau salturile bbox-ului galben.
AMBER_FALLBACK_MAX_Y_RATIO = 0.48
AMBER_FALLBACK_TRACK_CENTER_SCALE = 1.25
AMBER_FALLBACK_MAX_BBOX_WIDTH = 60
AMBER_FALLBACK_MAX_BBOX_HEIGHT = 120

def action_for_label(label):
    return ACTION_BY_LABEL.get(label, ACTION_BY_LABEL["unknown"])


def stabilize_detection_color(detection):
    global PENDING_COLOR_COUNT, PENDING_COLOR_LABEL, STABLE_COLOR_LABEL

    if detection is None:
        return None

    raw_label = detection["label"]
    detection["raw_label"] = raw_label

    if raw_label == STABLE_COLOR_LABEL:
        PENDING_COLOR_LABEL = None
        PENDING_COLOR_COUNT = 0
    else:
        if raw_label == PENDING_COLOR_LABEL:
            PENDING_COLOR_COUNT += 1
        else:
            PENDING_COLOR_LABEL = raw_label
            PENDING_COLOR_COUNT = 1

        if raw_label == "red":
            confirm_frames = RED_CONFIRM_FRAMES
        elif raw_label == "yellow":
            # Galben inseamna SLOW si este starea mai sigura. LED-ul ambre
            # apare uneori galben doar intermitent din cauza expunerii ZED.
            confirm_frames = YELLOW_CONFIRM_FRAMES
        else:
            confirm_frames = COLOR_CONFIRM_FRAMES

        if PENDING_COLOR_COUNT >= confirm_frames:
            STABLE_COLOR_LABEL = raw_label
            PENDING_COLOR_LABEL = None
            PENDING_COLOR_COUNT = 0

    stable_label = STABLE_COLOR_LABEL or "unknown"
    detection["label"] = stable_label
    detection["action"] = action_for_label(stable_label)
    return detection


def reset_color_filter():
    global PENDING_COLOR_COUNT, PENDING_COLOR_LABEL, STABLE_COLOR_LABEL

    STABLE_COLOR_LABEL = None
    PENDING_COLOR_LABEL = None
    PENDING_COLOR_COUNT = 0


def proximity_from_distance(distance_m, near_threshold, far_threshold):
    if distance_m is None:
        return "unknown"

    if distance_m < near_threshold:
        return "aproape"

    if distance_m > far_threshold:
        return "departe"

    return "mediu"

def proximity_from_bbox_height(bbox_height, near_height, far_height):
    if bbox_height is None:
        return "unknown"

    if bbox_height >= near_height:
        return "aproape"

    if bbox_height <= far_height:
        return "departe"

    return "mediu"

def estimate_distance_from_bbox(bbox, bbox_distance_k, bbox_distance_offset):
    x1, y1, x2, y2 = bbox

    box_h = y2 - y1

    if box_h <= 0:
        return None

    distance_m = bbox_distance_k / box_h + bbox_distance_offset

    return float(max(0.10, distance_m))

def estimate_distance_from_depth(depth_frame, bbox, lamp_y_ratio=None):
    if depth_frame is None:
        return None

    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return None

    # La fel ca pentru semne: ignoram marginile bbox-ului, unde apar frecvent
    # fundalul si gaurile din harta de adancime ZED.
    inset_x = int(box_w * DISTANCE_BOX_MARGIN_RATIO)
    inset_y = int(box_h * DISTANCE_BOX_MARGIN_RATIO)

    cx1 = x1 + inset_x
    cx2 = x2 - inset_x

    # Cand stim pozitia becului aprins, masuram in jurul lui. Pentru verde,
    # de exemplu, becul este jos (lamp_y_ratio ~ 0.90), iar centrul geometric
    # al bbox-ului poate contine numai carcasa neagra si fundalul din spate.
    if lamp_y_ratio is not None and 0.0 <= lamp_y_ratio <= 1.0:
        lamp_y = y1 + int(box_h * lamp_y_ratio)
        lamp_half_height = max(4, int(box_h * 0.18))
        cy1 = max(y1, lamp_y - lamp_half_height)
        cy2 = min(y2, lamp_y + lamp_half_height)
    else:
        cy1 = y1 + inset_y
        cy2 = y2 - inset_y

    if cx2 <= cx1 or cy2 <= cy1:
        cx1, cy1, cx2, cy2 = x1, y1, x2, y2

    def valid_values(region):
        return region[
            np.isfinite(region)
            & (region >= ZED_DEPTH_MIN_M)
            & (region < 5.0)
        ]

    # Bbox-ul vine din imaginea LEFT ZED completa, aceleasi coordonate ca
    # depth_frame. Crop-ul RGB pentru culoare nu schimba coordonatele.
    depth_crop = depth_frame[cy1:cy2, cx1:cx2]
    valid_depth = valid_values(depth_crop)

    # LED-urile foarte luminoase pot lasa goluri in depth. In cazul acesta
    # revenim la zona centrala veche, in loc sa producem o distanta aleatoare.
    if valid_depth.size < 20 and lamp_y_ratio is not None:
        cy1 = y1 + inset_y
        cy2 = y2 - inset_y
        depth_crop = depth_frame[cy1:cy2, cx1:cx2]
        valid_depth = valid_values(depth_crop)

    # Nu acceptam o distanta calculata din doar cativa pixeli ZED valizi.
    if valid_depth.size < 20:
        return None

    # Comportamentul vechi: eliminam extremele spatiale inainte de mediana.
    p20 = np.percentile(valid_depth, 20)
    p80 = np.percentile(valid_depth, 80)
    filtered_depth = valid_depth[
        (valid_depth >= p20) & (valid_depth <= p80)
    ]

    if filtered_depth.size < 10:
        return None

    # La apropiere, carcasa ocupa mult din imagine, dar LED-ul saturat poate
    # lasa fundalul sa domine numeric. Quartila inferioara selecteaza suprafata
    # apropiata. Pentru bbox-urile normale pastram exact mediana veche.
    if box_h >= CLOSE_DISTANCE_BBOX_HEIGHT and lamp_y_ratio is not None:
        return float(np.percentile(valid_depth, 25))

    return float(np.median(filtered_depth))


def correct_very_close_distance(zed_distance_m, bbox_height):
    """Foloseste perspectiva numai cand ZED nu mai urmareste apropierea."""
    global CLOSE_DISTANCE_ANCHOR_BBOX_HEIGHT, CLOSE_DISTANCE_ANCHOR_M
    global CLOSE_DISTANCE_MODE_ACTIVE

    if bbox_height is None or bbox_height <= 0:
        return zed_distance_m, "zed"

    # Ancora se actualizeaza numai in zona in care semaforul incape bine in
    # imagine. Nu o suprascriem cu valoarea ZED blocata din apropiere.
    if (
        not CLOSE_DISTANCE_MODE_ACTIVE
        and zed_distance_m is not None
        and bbox_height <= CLOSE_DISTANCE_ANCHOR_MAX_BBOX_HEIGHT
        and zed_distance_m >= ZED_DEPTH_MIN_M
        and zed_distance_m <= 1.50
    ):
        CLOSE_DISTANCE_ANCHOR_M = float(zed_distance_m)
        CLOSE_DISTANCE_ANCHOR_BBOX_HEIGHT = float(bbox_height)
        return float(zed_distance_m), "zed"

    if (
        CLOSE_DISTANCE_ANCHOR_M is None
        or CLOSE_DISTANCE_ANCHOR_BBOX_HEIGHT is None
    ):
        return zed_distance_m, "zed"

    # Iesim din modul apropiat numai cand bbox-ul revine clar in zona in care
    # ZED poate furniza din nou o ancora. O variatie 100 -> 83 px nu trebuie
    # sa reactiveze citirea ZED blocata la 0.45...0.60 m.
    if (
        CLOSE_DISTANCE_MODE_ACTIVE
        and bbox_height <= CLOSE_DISTANCE_ANCHOR_MAX_BBOX_HEIGHT
        and zed_distance_m is not None
        and zed_distance_m <= 1.50
    ):
        CLOSE_DISTANCE_MODE_ACTIVE = False
        CLOSE_DISTANCE_ANCHOR_M = float(zed_distance_m)
        CLOSE_DISTANCE_ANCHOR_BBOX_HEIGHT = float(bbox_height)
        return float(zed_distance_m), "zed"

    if (
        not CLOSE_DISTANCE_MODE_ACTIVE
        and bbox_height < CLOSE_DISTANCE_SWITCH_BBOX_HEIGHT
    ):
        return zed_distance_m, "zed"

    perspective_distance = (
        CLOSE_DISTANCE_ANCHOR_M
        * CLOSE_DISTANCE_ANCHOR_BBOX_HEIGHT
        / float(bbox_height)
    )
    perspective_distance = float(max(0.10, perspective_distance))

    # Daca ZED scade impreuna cu bbox-ul, el ramane sursa mai buna. Trecem
    # la estimarea relativa doar cand citirea ZED este absenta sau ramane
    # vizibil mai mare decat indica marirea carcasei.
    if (
        zed_distance_m is None
        or zed_distance_m > perspective_distance + CLOSE_DISTANCE_ZED_TOLERANCE_M
    ):
        CLOSE_DISTANCE_MODE_ACTIVE = True
        return perspective_distance, "bbox_close"

    return float(zed_distance_m), "zed"

def format_distance(distance_m):
    if distance_m is None:
        return "DIST: ?"

    return f"DIST: {distance_m:.2f}m"

def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_w * intersection_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection

    return intersection / union if union > 0 else 0.0


def make_display_bbox(raw_bbox, frame_shape, final_label, lamp_y_ratio):
    """Mareste numai dreptunghiul desenat, fara sa afecteze ZED/tracking."""
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = raw_bbox
    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = max(2, int(box_w * 0.10))
    pad_top = max(2, int(box_h * 0.08))
    pad_bottom = max(2, int(box_h * 0.06))

    # Cand rosul este foarte intens, YOLO incadreaza uneori numai carcasa de
    # sub LED. Extindem atunci dreptunghiul in sus ca sa includa tot corpul.
    if final_label == "red":
        # Chiar si cand LED-ul este vizibil, lasam putin mai mult spatiu sus
        # pentru haloul intens si marginea superioara a carcasei.
        pad_top = max(pad_top, int(box_h * 0.15))
        if lamp_y_ratio is None or lamp_y_ratio > 0.32:
            pad_top = max(pad_top, int(box_h * 0.35))

    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_top),
        min(frame_w - 1, x2 + pad_x),
        min(frame_h - 1, y2 + pad_bottom),
    )


def combine_color_and_position(yolo_label, crop_label, crop_confidence, lamp_position):
    """
    Combina YOLO, culoarea HSV si pozitia becului.

    Pozitia este un vot secundar. Nu mai fortam rosu la orice conflict,
    deoarece un bbox incomplet poate face pozitia nesigura si transforma
    galbenul sau verdele in rosu.
    """
    valid = {"red", "yellow", "green"}

    yolo_ok = yolo_label in valid
    crop_ok = (
        crop_label in valid
        and crop_confidence >= COLOR_OVERRIDE_MIN_CONFIDENCE
    )
    position_ok = lamp_position in valid

    # Doua surse independente sunt de acord.
    if yolo_ok and crop_ok and yolo_label == crop_label:
        return crop_label, "yolo_color_agree"

    # Modelul YOLO confunda sistematic LED-ul ambre cu red_traffic_light, iar
    # pozitia aparenta sare cand bbox-ul se scurteaza. Dupa separarea HSV+BGR
    # calibrata (rosu G/R < 0.25, portocaliu G/R ~0.44), un crop galben cu
    # incredere mare este mai sigur decat clasa YOLO si pozitia instabila.
    if crop_label == "yellow" and crop_confidence >= 0.60:
        return "yellow", "strong_amber_color"

    if crop_ok and position_ok and crop_label == lamp_position:
        return crop_label, "color_position_agree"

    if yolo_ok and position_ok and yolo_label == lamp_position:
        return yolo_label, "yolo_position_agree"

    # Culoarea puternica din crop poate corecta clasa YOLO.
    if crop_ok and crop_confidence >= 0.45:
        return crop_label, "strong_color"

    # Daca pozitia confirma o culoare moderata, acceptam culoarea.
    if crop_ok and position_ok and crop_confidence >= 0.25:
        return crop_label, "color_with_position"

    # YOLO ramane fallback-ul principal cand HSV este nesigur.
    if yolo_ok:
        return yolo_label, "yolo_fallback"

    if crop_ok:
        return crop_label, "color_fallback"

    return "unknown", "unresolved"

def smooth_distance(distance_m, bbox=None):
    global DISTANCE_TRACK_BBOX, SMOOTHED_DISTANCE_M

    if bbox is not None:
        # Daca bbox-ul sare intr-o alta zona, este aproape sigur alta detectie.
        # Resetam istoricul ca distanta vechiului obiect sa nu fie amestecata
        # cu distanta noului semafor.
        if DISTANCE_TRACK_BBOX is not None:
            iou = bbox_iou(DISTANCE_TRACK_BBOX, bbox)
            x1, y1, x2, y2 = bbox
            px1, py1, px2, py2 = DISTANCE_TRACK_BBOX
            center_distance = np.hypot(
                (x1 + x2 - px1 - px2) * 0.5,
                (y1 + y2 - py1 - py2) * 0.5,
            )
            scale = max(
                x2 - x1, y2 - y1,
                px2 - px1, py2 - py1,
                1,
            )
            if iou < 0.02 and center_distance > 2.0 * scale:
                DISTANCE_HISTORY.clear()
                SMOOTHED_DISTANCE_M = None

        DISTANCE_TRACK_BBOX = bbox

    if distance_m is None:
        # Ca in trackerul semnelor, o citire lipsa pastreaza ultima distanta.
        return SMOOTHED_DISTANCE_M

    # Mediana ultimelor cinci citiri, apoi netezire exponentiala. Nu schimbam
    # istoricul din cauza unei singure valori ZED aberante.
    DISTANCE_HISTORY.append(distance_m)
    filtered_distance = float(np.median(DISTANCE_HISTORY))

    if SMOOTHED_DISTANCE_M is None:
        SMOOTHED_DISTANCE_M = filtered_distance
    else:
        confirmed_close = sum(
            value <= CLOSE_DISTANCE_CONFIRM_M for value in DISTANCE_HISTORY
        ) >= CLOSE_DISTANCE_CONFIRM_FRAMES
        if (
            confirmed_close
            and filtered_distance <= CLOSE_DISTANCE_CONFIRM_M
            and SMOOTHED_DISTANCE_M - filtered_distance > 0.25
        ):
            # Mediana istoriei confirma apropierea; nu mai pastram timp
            # indelungat distanta veche de 0.5...1.3 m.
            SMOOTHED_DISTANCE_M = filtered_distance
        else:
            SMOOTHED_DISTANCE_M += DISTANCE_SMOOTHING_ALPHA * (
                filtered_distance - SMOOTHED_DISTANCE_M
            )

    return float(SMOOTHED_DISTANCE_M)


def reset_distance_filter():
    global CLOSE_DISTANCE_MODE_ACTIVE
    global DISTANCE_TRACK_BBOX, SMOOTHED_DISTANCE_M

    DISTANCE_HISTORY.clear()
    SMOOTHED_DISTANCE_M = None
    DISTANCE_TRACK_BBOX = None
    # Ancora apartine geometriei camerei si aceleiasi carcase, nu track-ului
    # temporar. O pastram daca YOLO pierde cateva cadre la apropiere.
    CLOSE_DISTANCE_MODE_ACTIVE = False


def infer_color_from_crop(crop, color_ratio_threshold, yolo_hint="unknown"):
    if crop is None or crop.size == 0:
        return "unknown", 0.0, "unknown", None

    # Marim mai mult crop-urile semafoarelor indepartate. Interpolarea cubica
    # pastreaza mai bine pata luminoasa decat cea liniara pe bbox-uri de doar
    # cateva zeci de pixeli.
    crop_h, crop_w = crop.shape[:2]
    upscale = 5.0 if max(crop_w, crop_h) < 60 else 3.0
    crop = cv2.resize(
        crop,
        None,
        fx=upscale,
        fy=upscale,
        interpolation=cv2.INTER_CUBIC,
    )

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h, w = hsv.shape[:2]

    # Scoatem marginile ca să ignorăm rama semaforului / fundalul.
    margin_x = int(w * 0.10)
    margin_y = int(h * 0.05)

    hsv_inner = hsv[
        margin_y : h - margin_y if h - margin_y > margin_y else h,
        margin_x : w - margin_x if w - margin_x > margin_x else w,
    ]

    if hsv_inner.size == 0:
        hsv_inner = hsv

    saturation = hsv_inner[:, :, 1]
    value = hsv_inner[:, :, 2]

    # Corpul semaforului si LED-urile stinse pot avea o tenta colorata care
    # domina numeric crop-ul. Analizam numai partea cea mai luminoasa, adica
    # LED-ul aprins si haloul lui. Pragul relativ se adapteaza expunerii ZED.
    saturated_values = value[saturation > 30]
    if saturated_values.size:
        relative_brightness = max(
            70.0,
            float(np.percentile(saturated_values, 70)),
            float(np.max(saturated_values)) * 0.55,
        )
    else:
        relative_brightness = 70.0

    bright_mask = (
        (saturation > 30) & (value >= relative_brightness)
    ).astype(np.uint8) * 255

    # Bbox-ul YOLO poate contine carcasa, alte LED-uri si lumini din fundal.
    # Pastram o singura componenta luminoasa: cea cu intensitatea cea mai
    # mare, favorizand moderat si o pata coerenta fata de un pixel izolat.
    component_kernel = np.ones((3, 3), np.uint8)
    bright_mask = cv2.morphologyEx(
        bright_mask,
        cv2.MORPH_CLOSE,
        component_kernel,
    )
    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(bright_mask)
    )

    best_component = None
    best_component_score = -1.0
    for component_id in range(1, component_count):
        area = int(component_stats[component_id, cv2.CC_STAT_AREA])
        if area < 4:
            continue

        component_pixels = component_labels == component_id
        component_value = value[component_pixels]
        component_score = float(np.percentile(component_value, 90)) * (
            1.0 + 0.05 * np.sqrt(area)
        )
        if component_score > best_component_score:
            best_component = component_id
            best_component_score = component_score

    if best_component is not None:
        bright_mask = (
            component_labels == best_component
        ).astype(np.uint8) * 255

    # Pozitia becului aprins in corpul vertical al semaforului este un vot
    # independent de culoare: sus=rosu, mijloc=galben, jos=verde. Il folosim
    # numai cand componenta luminoasa ocupa o parte suficient de mica dintr-un
    # crop vertical; daca YOLO a incadrat doar becul, pozitia nu este sigura.
    lamp_position = "unknown"
    lamp_y_ratio = None
    if best_component is not None:
        component_y = int(
            component_stats[best_component, cv2.CC_STAT_TOP]
        )
        component_h = int(
            component_stats[best_component, cv2.CC_STAT_HEIGHT]
        )
        inner_h, inner_w = hsv_inner.shape[:2]
        component_height_ratio = component_h / float(max(inner_h, 1))
        crop_vertical_ratio = inner_h / float(max(inner_w, 1))
        lamp_y_ratio = (component_y + component_h * 0.5) / float(
            max(inner_h, 1)
        )

        if crop_vertical_ratio >= 1.45 and component_height_ratio <= 0.45:
            if lamp_y_ratio < 0.38:
                lamp_position = "red"
            elif lamp_y_ratio > 0.62:
                lamp_position = "green"
            else:
                lamp_position = "yellow"

    active_pixels = max(1, cv2.countNonZero(bright_mask))

    scores = {}

    for label, ranges in COLOR_RANGES.items():
        color_mask = np.zeros(hsv_inner.shape[:2], dtype=np.uint8)

        for lower, upper in ranges:
            current_mask = cv2.inRange(
                hsv_inner,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            )
            color_mask = cv2.bitwise_or(color_mask, current_mask)

        color_mask = cv2.bitwise_and(color_mask, bright_mask)

        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        color_pixels = cv2.countNonZero(color_mask)
        score = color_pixels / float(active_pixels)

        scores[label] = score

    red_score = scores.get("red", 0.0)
    yellow_score = scores.get("yellow", 0.0)
    green_score = scores.get("green", 0.0)

    # Folosim si nuanta mediana a componentei luminoase. La distanta,
    # galbenul ambre isi pierde saturatia, dar ramane de regula sub H=36.
    active_hues = hsv_inner[:, :, 0][bright_mask > 0]
    median_hue = float(np.median(active_hues)) if active_hues.size else None
    green_red_ratio = None
    if active_hues.size:
        bgr_inner = cv2.cvtColor(hsv_inner, cv2.COLOR_HSV2BGR)
        active_bgr = bgr_inner[bright_mask > 0]
        median_bgr = np.median(active_bgr, axis=0)
        green_red_ratio = float(median_bgr[1]) / max(1.0, float(median_bgr[2]))

    if red_score >= 0.08 and red_score >= max(yellow_score, green_score) * 0.90:
        # Camera ZED poate impinge LED-ul ambre pana in H=0..4, adica exact
        # in masca rosie. Il separam prin pozitia becului si raportul G/R:
        # rosul real are putin verde, iar portocaliul pastreaza mai mult verde.
        amber_in_red_range = (
            green_red_ratio is not None
            and (
                (lamp_position == "yellow" and green_red_ratio >= 0.25)
                or (
                    lamp_position == "unknown"
                    and green_red_ratio >= 0.25
                )
            )
        )
        if amber_in_red_range:
            return (
                "yellow",
                max(red_score, yellow_score),
                lamp_position,
                lamp_y_ratio,
            )

        return "red", red_score, lamp_position, lamp_y_ratio

    # Un nucleu rosu supraexpus poate cadea exact la H=5 si intra in prima
    # treapta a mastii portocalii. Raportul canalelor il separa clar in
    # capturile ZED: rosu ~0.17, portocaliu ~0.44.
    if (
        green_red_ratio is not None
        and green_red_ratio < 0.25
        and max(red_score, yellow_score) >= 0.06
    ):
        return (
            "red",
            max(red_score, yellow_score),
            lamp_position,
            lamp_y_ratio,
        )

    # Verificarea geometrica este decisiva pentru verdele deplasat cromatic
    # spre galben de white balance: becul este in treimea de jos, iar canalul
    # verde ramane suficient de puternic fata de rosu. In log, toate cadrele
    # confundate aveau lamp_y_ratio=0.80...0.97.
    if (
        lamp_position == "green"
        and green_red_ratio is not None
        and green_red_ratio >= 0.75
    ):
        return (
            "green",
            max(green_score, yellow_score),
            lamp_position,
            lamp_y_ratio,
        )

    if median_hue is not None and 28.0 <= median_hue < 36.0:
        # Vot majoritar in zona ambigua: raport cromatic, YOLO si pozitia
        # becului. Astfel, un verde indepartat aflat jos nu devine galben doar
        # pentru ca white balance-ul i-a coborat Hue-ul.
        color_vote = (
            "green"
            if green_red_ratio is not None and green_red_ratio > 1.05
            else "yellow"
        )
        votes = [color_vote]
        if yolo_hint in ("yellow", "green"):
            votes.append(yolo_hint)
        if lamp_position in ("yellow", "green"):
            votes.append(lamp_position)

        green_votes = votes.count("green")
        yellow_votes = votes.count("yellow")
        if green_votes > yellow_votes:
            return (
                "green",
                max(green_score, yellow_score),
                lamp_position,
                lamp_y_ratio,
            )
        if yellow_score >= 0.06:
            return "yellow", yellow_score, lamp_position, lamp_y_ratio

    if (
        yellow_score >= 0.06
        and median_hue is not None
        and median_hue < 28.0
        and yellow_score >= green_score * 0.70
    ):
        return "yellow", yellow_score, lamp_position, lamp_y_ratio

    if (
        green_score >= 0.10
        and median_hue is not None
        and median_hue >= 36.0
        and green_score > yellow_score * 0.85
    ):
        return "green", green_score, lamp_position, lamp_y_ratio

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score < 0.07:
        return "unknown", best_score, lamp_position, lamp_y_ratio

    return best_label, best_score, lamp_position, lamp_y_ratio


def detect_amber_fallback(frame, depth_frame, args):
    """Detecteaza direct pata luminoasa galben-portocalie cand YOLO nu
    propune niciun bbox util. Este un fallback, nu inlocuieste YOLO."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    frame_h, frame_w = frame.shape[:2]

    mask = cv2.inRange(
        hsv,
        np.array(
            (AMBER_FALLBACK_H_MIN, AMBER_FALLBACK_S_MIN, AMBER_FALLBACK_V_MIN),
            dtype=np.uint8,
        ),
        np.array((AMBER_FALLBACK_H_MAX, 255, 255), dtype=np.uint8),
    )

    # Pentru masina ne intereseaza zona centrala; marginile contin monitoare,
    # reflexii si alte surse luminoase.
    allowed = np.zeros_like(mask)
    margin_x = int(frame_w * AMBER_FALLBACK_HORIZONTAL_MARGIN_RATIO)
    allowed[
        0:int(frame_h * AMBER_FALLBACK_MAX_Y_RATIO),
        margin_x:frame_w - margin_x,
    ] = 255
    mask = cv2.bitwise_and(mask, allowed)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None

    for component_id in range(1, count):
        x, y, w, h, area = stats[component_id]
        if area < AMBER_FALLBACK_MIN_AREA:
            continue
        if area > AMBER_FALLBACK_MAX_AREA_RATIO * frame_w * frame_h:
            continue

        aspect = w / float(max(h, 1))
        if not 0.35 <= aspect <= 2.8:
            continue

        # Daca avem deja un semafor urmarit, fallback-ul ambre trebuie sa fie
        # in aceeasi zona. Asa nu sare pe LED-uri/reflexii portocalii din alta
        # parte a cadrului si nu strica nici distanta.
        if DISTANCE_TRACK_BBOX is not None:
            tx1, ty1, tx2, ty2 = DISTANCE_TRACK_BBOX
            tcx, tcy = (tx1 + tx2) * 0.5, (ty1 + ty2) * 0.5
            ccx, ccy = x + w * 0.5, y + h * 0.5
            # Folosim numai dimensiunea track-ului anterior. Daca includem
            # candidatul curent in scala, un halou/bbox mare face pragul atat
            # de permisiv incat urmatorul LED de pe masa este acceptat.
            track_scale = max(tx2 - tx1, ty2 - ty1, 1)
            max_center_distance = max(
                20.0,
                AMBER_FALLBACK_TRACK_CENTER_SCALE * track_scale,
            )
            if np.hypot(ccx - tcx, ccy - tcy) > max_center_distance:
                continue

        pixels = labels == component_id
        mean_h = float(np.mean(h_ch[pixels]))
        mean_v = float(np.mean(v_ch[pixels]))
        mean_s = float(np.mean(s_ch[pixels]))
        mean_bgr = np.mean(frame[pixels], axis=0)
        green_red_ratio = float(mean_bgr[1]) / max(1.0, float(mean_bgr[2]))

        # Rosul foarte intens produce un halou portocaliu. Daca ultima
        # detectie YOLO sigura era rosie si componenta este in zona de sus a
        # aceleiasi carcase, nu o reinterpretam ca galben. Un galben real se
        # afla mai jos, aproximativ in treimea din mijloc.
        if (
            LAST_RELIABLE_YOLO_LABEL == "red"
            and LAST_RELIABLE_YOLO_BBOX is not None
        ):
            lx1, ly1, lx2, ly2 = LAST_RELIABLE_YOLO_BBOX
            last_w = max(1, lx2 - lx1)
            last_h = max(1, ly2 - ly1)
            component_cx = x + w * 0.5
            component_cy = y + h * 0.5
            same_horizontal_area = (
                lx1 - 0.75 * last_w
                <= component_cx
                <= lx2 + 0.75 * last_w
            )
            red_lamp_area = component_cy <= ly1 + 0.45 * last_h
            if same_horizontal_area and red_lamp_area:
                continue

        # Separare masurata din capturi: rosul are G/R ~0.17, iar LED-ul
        # portocaliu are G/R ~0.44. Nu acceptam haloul rosu drept ambre.
        if green_red_ratio < 0.25:
            continue

        # H=28..35 poate fi fie galben, fie verde indepartat. Verdele are de
        # regula G > R, iar daca track-ul stabil este deja verde nu permitem
        # unui singur fallback ambiguu sa-l transforme in galben.
        if mean_h >= 28.0 and green_red_ratio > 1.05:
            continue
        if STABLE_COLOR_LABEL == "green" and mean_h >= 26.0:
            continue

        circularity_score = min(aspect, 1.0 / max(aspect, 1e-6))
        score = (
            area / float(max(w * h, 1))
            * (mean_v / 255.0)
            * (mean_s / 255.0)
            * circularity_score
        )

        if score < AMBER_FALLBACK_MIN_SCORE:
            continue
        if best is None or score > best["score"]:
            best = {"x": x, "y": y, "w": w, "h": h, "score": score}

    if best is None:
        return None

    # Extindem pata luminoasa ca sa obtinem un bbox vizibil pentru semafor,
    # dar distanta se citeste din zona LED-ului, nu din fundal.
    x, y, w, h = best["x"], best["y"], best["w"], best["h"]
    cx, cy = x + w // 2, y + h // 2
    bbox_w = min(
        AMBER_FALLBACK_MAX_BBOX_WIDTH,
        max(24, int(w * 2.5)),
    )
    bbox_h = min(
        AMBER_FALLBACK_MAX_BBOX_HEIGHT,
        max(72, int(h * 7.0), int(bbox_w * 2.5)),
    )
    x1 = max(0, cx - bbox_w // 2)
    x2 = min(frame_w - 1, cx + bbox_w // 2)
    y1 = max(0, cy - bbox_h // 2)
    y2 = min(frame_h - 1, cy + bbox_h // 2)
    bbox = (x1, y1, x2, y2)

    led_pad = 2
    led_bbox = (
        max(0, x - led_pad),
        max(0, y - led_pad),
        min(frame_w - 1, x + w + led_pad),
        min(frame_h - 1, y + h + led_pad),
    )
    zed_distance_m = estimate_distance_from_depth(depth_frame, led_bbox)
    bbox_distance_m = estimate_distance_from_bbox(
        bbox, args.bbox_distance_k, args.bbox_distance_offset
    )

    # Bbox-ul fallback este construit artificial in jurul LED-ului, deci
    # formula bbox nu reprezinta o distanta reala. Folosim numai ZED; daca
    # ZED nu are valori valide, pastram distanta anterioara prin smoothing.
    if zed_distance_m is not None:
        distance_m, source = zed_distance_m, "zed"
    else:
        distance_m, source = None, "held"

    return {
        "label": "yellow",
        "action": action_for_label("yellow"),
        "confidence": float(best["score"]),
        "color_confidence": float(best["score"]),
        "bbox": bbox,
        # Este o estimare vizuala a carcasei. Nu pornim tracking-ul si nu
        # calculam distanta din ea, fiindca nu provine din YOLO.
        "draw_bbox": True,
        "trackable": False,
        "bbox_height": y2 - y1,
        "class_name": "amber_hsv_fallback",
        "yolo_label": "unknown",
        "crop_label": "yellow",
        # Bbox-ul fallback este construit in jurul petei, deci pozitia
        # relativa in el este artificial centrata si nu constituie un vot.
        "lamp_position": "unknown",
        "lamp_y_ratio": None,
        "score": 2.0 + float(best["score"]),
        "distance_m": distance_m,
        "raw_distance_m": distance_m,
        "bbox_distance_m": bbox_distance_m,
        "zed_distance_m": zed_distance_m,
        "distance_source": source,
        "proximity": proximity_from_distance(
            distance_m, args.near_distance, args.far_distance
        ),
    }


def normalize_yolo_label(class_name):
    class_name = class_name.lower().strip()

    if "red" in class_name:
        return "red"
    if "yellow" in class_name:
        return "yellow"
    if "green" in class_name:
        return "green"

    return "unknown"


def resolve_model_path(configured_path):
    if configured_path:
        model_path = Path(configured_path).expanduser()

        if model_path.exists():
            return str(model_path)

        raise FileNotFoundError(f"Model not found: {model_path}")

    project_root = Path(__file__).resolve().parents[3]

    search_dirs = [
        Path("/home/wsadmin/Desktop/Rc-Car-Chimera/Rc-Car-Chimera/Rc-Car/src/model"),
        project_root / "src" / "model",
        Path.cwd() / "src" / "model",
        Path(__file__).resolve().parents[2] / "model",
    ]

    for search_dir in search_dirs:
        for model_file_name in MODEL_FILE_NAMES:
            candidate = search_dir / model_file_name

            if candidate.exists():
                return str(candidate)

    raise FileNotFoundError(
        "Traffic light Keras model was not found. "
        "Use --model-path with the full path of your .h5 model."
    )


def resolve_yolo_model_path(configured_path):
    model_path = Path(configured_path).expanduser()

    if model_path.exists() and model_path.is_file():
        return str(model_path)

    raise FileNotFoundError(
        "YOLO model was not found. Expected file:\n"
        f"{model_path}\n\n"
        "Important: use the .pt file directly. Do not extract/unzip best.pt."
    )


def preprocess(frame, input_size):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_size)
    image = image.astype(np.float32)
    return np.expand_dims(image, axis=0)


def predict_label(model, frame, threshold, input_size):
    predictions = model.predict(preprocess(frame, input_size), verbose=0)
    probabilities = predictions[0]

    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])

    if class_index >= len(LABELS):
        return "unknown", confidence, probabilities, class_index

    label = LABELS[class_index]

    if confidence < threshold:
        return "unknown", confidence, probabilities, class_index

    return label, confidence, probabilities, class_index


def make_one_hot(class_index, size):
    return [1 if i == class_index else 0 for i in range(size)]


def print_prediction(label, confidence, probabilities, class_index):
    one_hot = make_one_hot(class_index, len(LABELS))

    print("\n==============================")
    print("SCAN RESULT")
    print("==============================")
    print(f"Detected label: {label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Class index: {class_index}")
    print(f"One-hot: {one_hot}")

    print("\nProbabilities:")
    for i, label_name in enumerate(LABELS):
        value = float(probabilities[i]) if i < len(probabilities) else 0.0
        print(f"  {i} - {label_name}: {value:.4f}")

    print("==============================\n")


def crop_center(frame, scale):
    h, w = frame.shape[:2]

    crop_w = int(w * scale)
    crop_h = int(h * scale)

    x1 = max(0, (w - crop_w) // 2)
    y1 = max(0, (h - crop_h) // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    crop = frame[y1:y2, x1:x2]

    return crop, (x1, y1, x2, y2)


def get_roi(frame, args):
    if args.no_crop or args.roi_scale >= 1.0:
        h, w = frame.shape[:2]
        return frame, (0, 0, w, h)

    if args.roi is not None:
        x, y, w, h = args.roi

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)

        crop = frame[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2)

    return crop_center(frame, args.roi_scale)


def draw_preview(frame, args):
    preview = frame.copy()

    _, roi_box = get_roi(preview, args)
    x1, y1, x2, y2 = roi_box

    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 90), (25, 25, 25), -1)

    cv2.putText(
        preview,
        "Press S to scan | Press Q to quit",
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )

    return preview


def draw_result(frame, roi_box, label, confidence, probabilities):
    x1, y1, x2, y2 = roi_box

    color = DISPLAY_COLORS.get(label, DISPLAY_COLORS["unknown"])

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (25, 25, 25), -1)

    cv2.putText(
        frame,
        f"{label.upper()}  {confidence:.2f}",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        color,
        3,
        cv2.LINE_AA,
    )

    y = 130

    for index, label_name in enumerate(LABELS):
        value = float(probabilities[index]) if index < len(probabilities) else 0.0

        cv2.putText(
            frame,
            f"{label_name}: {value:.2f}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            DISPLAY_COLORS.get(label_name, DISPLAY_COLORS["unknown"]),
            2,
            cv2.LINE_AA,
        )

        y += 30


def class_name_from_yolo(model, class_index):
    names = model.names

    if isinstance(names, dict):
        return str(names.get(class_index, class_index)).lower()

    return str(names[class_index]).lower()


def detect_yolo_traffic_light(model, frame, args, depth_frame=None):
    global LAST_RELIABLE_YOLO_BBOX, LAST_RELIABLE_YOLO_LABEL

    results = model.predict(
        frame,
        conf=args.yolo_conf,
        imgsz=args.yolo_imgsz,
        device="cpu",
        verbose=False,
    )

    if not results:
        if DISTANCE_TRACK_BBOX is None:
            return detect_amber_fallback(frame, depth_frame, args)
        return None

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        if DISTANCE_TRACK_BBOX is None:
            return detect_amber_fallback(frame, depth_frame, args)
        return None

    valid_classes = {
        item.strip().lower()
        for item in args.yolo_valid_classes.split(",")
        if item.strip()
    }

    best_detection = None
    frame_h, frame_w = frame.shape[:2]

    for box in result.boxes:
        class_index = int(box.cls[0])
        class_name = class_name_from_yolo(model, class_index)

        is_valid_class = (
            class_name in valid_classes
            or "traffic" in class_name
            or "red" in class_name
            or "yellow" in class_name
            or "green" in class_name
        )

        if not is_valid_class:
            continue

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]

        x1 = max(0, min(x1, frame_w - 1))
        x2 = max(0, min(x2, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        y2 = max(0, min(y2, frame_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue
        # Un semafor real poate intra partial prin marginea de sus cand camera
        # masinutei este aproape de el. Pastram numai filtrarea marginilor
        # laterale; y1 <= 5 elimina chiar semaforul din captura de test.
        if x1 <= 5 or x2 >= frame_w - 5:
            continue

        box_w = x2 - x1
        box_h = y2 - y1
        box_center_x = (x1 + x2) * 0.5

        # Modelul mai propune puncte luminoase aproape patrate (LED-uri,
        # reflexii, elemente de pe monitor) drept semafor. Un corp de semafor
        # valid din setul nostru este vertical, inclusiv cand este foarte mic.
        min_height_width_ratio = MIN_TRAFFIC_LIGHT_HEIGHT_WIDTH_RATIO
        if DISTANCE_TRACK_BBOX is not None and box_h >= 80:
            # La apropiere carcasa poate fi taiata de cadru sau YOLO poate
            # include haloul lateral al LED-ului. Track-ul existent impiedica
            # aceasta relaxare sa initieze detectii false noi.
            min_height_width_ratio = 0.75
        if box_h / float(box_w) < min_height_width_ratio:
            continue
        if box_h < MIN_TRAFFIC_LIGHT_BBOX_HEIGHT:
            continue
        if (
            box_h < MIN_NEW_TRAFFIC_LIGHT_BBOX_HEIGHT
            and DISTANCE_TRACK_BBOX is None
        ):
            # Un semafor deja urmarit poate deveni foarte mic cand se
            # indeparteaza. Nu initiem insa un track nou dintr-un punct de
            # 14...19 px, interval in care apar multe LED-uri false.
            continue

        # Pentru controlul masinii ne intereseaza semaforul din fata. Eliminam
        # marginile laterale, unde in test apar monitorul si reflexiile lui.
        horizontal_margin = frame_w * DETECTION_HORIZONTAL_MARGIN_RATIO
        if not horizontal_margin <= box_center_x <= frame_w - horizontal_margin:
            continue

        raw_bbox = (x1, y1, x2, y2)
        yolo_label = normalize_yolo_label(class_name)

        # Bbox-ul YOLO ramane sursa unica pentru desenare, tracking, culoare
        # si distanta. Nu il extindem pe baza culorii prezise: asta muta
        # dreptunghiul in fundal si face verificarea pozitiei circulara.
        bbox = raw_bbox
        px1, py1, px2, py2 = bbox
        crop = frame[py1:py2, px1:px2]

        (
            crop_label,
            crop_confidence,
            lamp_position,
            lamp_y_ratio,
        ) = infer_color_from_crop(
            crop,
            args.color_ratio_threshold,
            yolo_hint="unknown",
        )

        # Respingem predictia neconfirmata numai daca bbox-ul este foarte mic.
        # Filtrul anterior respingea si rosul real cand LED-ul intens satura
        # crop-ul HSV. Bbox-ul fals din log avea 6x16 px; un corp suficient de
        # inalt poate ramane valid chiar daca HSV este temporar necunoscut.
        if (
            yolo_label in {"red", "yellow"}
            and crop_label == "unknown"
            and lamp_position == "unknown"
            and (box_w < 10 or box_h < 35)
        ):
            continue

        # Un track nou nu porneste de la LED-urile aflate jos pe masa. In log,
        # falsul green/red avea centrul la y=0.58...0.63 din cadru si a
        # contaminat distanta cu valori de 2.5...3 m. Semaforul real din fata
        # este in jumatatea superioara in configuratia masinutei.
        box_center_y = (y1 + y2) * 0.5
        if (
            DISTANCE_TRACK_BBOX is None
            and box_center_y > frame_h * 0.55
        ):
            continue

        label, decision_reason = combine_color_and_position(
            yolo_label=yolo_label,
            crop_label=crop_label,
            crop_confidence=crop_confidence,
            lamp_position=lamp_position,
        )
        color_confidence = crop_confidence if crop_label != "unknown" else confidence

        if label == "unknown":
            continue

        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1
        box_center_x = (x1 + x2) * 0.5
        bbox_height = box_h

        # Acelasi bbox brut este folosit si pentru ZED, si pentru formula
        # bazata pe inaltime; valorile din log corespund dreptunghiului afisat.
        bbox_distance_m = estimate_distance_from_bbox(
            raw_bbox,
            args.bbox_distance_k,
            args.bbox_distance_offset,
        )

        zed_distance_m = estimate_distance_from_depth(
            depth_frame,
            raw_bbox,
            lamp_y_ratio=lamp_y_ratio,
        )

        distance_source = "unknown"

        if args.distance_method == "bbox":
            distance_m = bbox_distance_m
            distance_source = "bbox"
        elif args.distance_method == "zed":
            # Comportamentul original cu care verdele a functionat corect:
            # folosim direct harta ZED, apoi filtrarea temporala comuna.
            distance_m = zed_distance_m
            distance_source = "zed"
        else:
            zed_matches_bbox = (
                zed_distance_m is not None
                and bbox_distance_m is not None
                and 0.25 <= zed_distance_m <= 5.0
                and abs(zed_distance_m - bbox_distance_m) <= max(
                    0.20,
                    bbox_distance_m * 0.25,
                )
            )

            if zed_matches_bbox:
                distance_m = zed_distance_m
                distance_source = "zed"
            else:
                distance_m = bbox_distance_m
                distance_source = "bbox"

        raw_distance_m = distance_m

        if args.proximity_method == "bbox_height":
            proximity = proximity_from_bbox_height(
                bbox_height,
                args.near_bbox_height,
                args.far_bbox_height,
            )
        else:
            proximity = proximity_from_distance(
                distance_m,
                args.near_distance,
                args.far_distance,
            )

        bbox_area = box_w * box_h
        normalized_area = bbox_area / float(frame_w * frame_h)
        center_y = (y1 + y2) * 0.5

        # Aria normalizata simpla era aproape zero si nu influenta selectia.
        # Folosim radacina ariei pentru a favoriza corpul complet al
        # semaforului fata de un punct luminos mic, fara a elimina semafoarele
        # reale aflate la distanta.
        size_score = min(1.00, 8.0 * np.sqrt(normalized_area))

        # In cadrul masinutei, semaforul relevant este de regula sus si
        # aproape de axa camerei. LED-urile false de pe placile de pe masa
        # apar mai jos si lateral. Scorul este gradual, ca sa permita in
        # continuare deplasarea semaforului prin cadru.
        horizontal_center_score = max(
            0.0,
            1.0 - abs(box_center_x - frame_w * 0.5) / (frame_w * 0.5),
        )
        upper_frame_score = max(0.0, 1.0 - center_y / float(frame_h))
        position_score = 0.45 * horizontal_center_score + 0.55 * upper_frame_score
        lamp_validation_score = 0.0
        if lamp_position != "unknown":
            if lamp_position == label:
                lamp_validation_score = 0.30
            else:
                # Pozitia este doar verificare secundara; un crop strans poate
                # centra artificial becul, deci contradictia nu elimina box-ul.
                lamp_validation_score = -0.10
        tracking_score = 0.0
        raw_x1, raw_y1, raw_x2, raw_y2 = raw_bbox
        raw_center_x = (raw_x1 + raw_x2) * 0.5
        raw_center_y = (raw_y1 + raw_y2) * 0.5
        raw_w = raw_x2 - raw_x1
        raw_h = raw_y2 - raw_y1

        if DISTANCE_TRACK_BBOX is not None:
            tracking_iou = bbox_iou(DISTANCE_TRACK_BBOX, raw_bbox)
            px1, py1, px2, py2 = DISTANCE_TRACK_BBOX
            previous_center_x = (px1 + px2) * 0.5
            previous_center_y = (py1 + py2) * 0.5
            center_distance = np.hypot(
                raw_center_x - previous_center_x,
                raw_center_y - previous_center_y,
            )
            tracking_scale = max(
                raw_w,
                raw_h,
                px2 - px1,
                py2 - py1,
                1,
            )
            center_bonus = max(0.0, 1.0 - center_distance / (3.0 * tracking_scale))

            # Asociem detectia numai cu acelasi obiect urmarit; o lumina din
            # alta zona nu are voie sa inlocuiasca semaforul intr-un cadru.
            matches_track = (
                tracking_iou >= 0.05
                or center_distance <= 1.5 * tracking_scale
            )
            if matches_track:
                tracking_score = 2.0 * tracking_iou + center_bonus
            else:
                # Nu inlocuim semaforul urmarit cu un LED/reflexie/persoana
                # aparuta brusc in alta zona. Daca semaforul chiar dispare,
                # perioada de gratie expira si se permite o achizitie noua.
                continue

        score = (
            confidence
            + color_confidence
            + size_score
            + position_score
            + lamp_validation_score
            + tracking_score
        )

        display_bbox = make_display_bbox(
            raw_bbox,
            frame.shape,
            label,
            lamp_y_ratio,
        )

        if best_detection is None or score > best_detection["score"]:
            best_detection = {
                "label": label,
                "action": action_for_label(label),
                "confidence": confidence,
                "color_confidence": color_confidence,
                "bbox": display_bbox,
                "draw_bbox": True,
                "trackable": True,
                "raw_bbox": raw_bbox,
                "bbox_height": bbox_height,
                "class_name": class_name,
                "yolo_label": yolo_label,
                "crop_label": crop_label,
                "lamp_position": lamp_position,
                "lamp_y_ratio": lamp_y_ratio,
                "decision_reason": decision_reason,
                "score": score,
                "distance_m": distance_m,
                "raw_distance_m": raw_distance_m,
                "bbox_distance_m": bbox_distance_m,
                "zed_distance_m": zed_distance_m,
                "distance_source": distance_source,
                "proximity": proximity,
            }

    # Fallback-ul global pentru ambre este permis numai la achizitia initiala.
    # Cat timp exista un track, main() pastreaza ultima detectie valida; astfel
    # un LED portocaliu de pe masa nu poate inlocui semaforul urmarit.
    if best_detection is not None and best_detection.get("trackable", True):
        LAST_RELIABLE_YOLO_BBOX = best_detection.get(
            "raw_bbox", best_detection["bbox"]
        )
        LAST_RELIABLE_YOLO_LABEL = best_detection["label"]

    if best_detection is None and DISTANCE_TRACK_BBOX is None:
        best_detection = detect_amber_fallback(frame, depth_frame, args)

    # Filtrarea se aplica numai detectiei selectate. Astfel, daca YOLO gaseste
    # mai multe bbox-uri in acelasi cadru, distantele lor nu se amesteca in
    # acelasi istoric.
    if best_detection is not None:
        if best_detection.get("trackable", True):
            if args.distance_method == "zed":
                corrected_distance, corrected_source = correct_very_close_distance(
                    best_detection.get("zed_distance_m"),
                    best_detection.get("bbox_height"),
                )
                if corrected_source == "bbox_close":
                    best_detection["raw_distance_m"] = corrected_distance
                    best_detection["distance_source"] = corrected_source

            best_detection["distance_m"] = smooth_distance(
                best_detection["raw_distance_m"],
                best_detection.get("raw_bbox", best_detection["bbox"]),
            )

        if args.proximity_method == "distance":
            best_detection["proximity"] = proximity_from_distance(
                best_detection["distance_m"],
                args.near_distance,
                args.far_distance,
            )

    return best_detection

def draw_yolo_result(frame, detection):
    display = frame.copy()

    if detection is None:
        label = "unknown"
        color = DISPLAY_COLORS["unknown"]

        cv2.rectangle(display, (0, 0), (display.shape[1], 110), (25, 25, 25), -1)

        cv2.putText(
            display,
            "SEARCHING TRAFFIC LIGHT",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            "DIST: ?  UNKNOWN",
            (20, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

        return display

    label = detection["label"]
    confidence = detection["confidence"]
    action = detection["action"]
    color = DISPLAY_COLORS.get(label, DISPLAY_COLORS["unknown"])
    x1, y1, x2, y2 = detection["bbox"]

    distance_m = detection.get("distance_m")
    proximity = detection.get("proximity", "unknown")
    distance_text = format_distance(distance_m)

    if detection.get("draw_bbox", True):
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)

    # Nu mai desenam al doilea dreptunghi si liniile de treimi. Ele erau utile
    # la debug, dar faceau afisarea confuza si pareau cadre gresite.

    cv2.rectangle(display, (0, 0), (display.shape[1], 110), (25, 25, 25), -1)

    cv2.putText(
        display,
        f"{label.upper()}  {confidence:.2f}  {action}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        f"{distance_text}  {proximity.upper()}",
        (20, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    return display


def print_yolo_detection(detection, previous_state):
    if detection is None:
        label = "unknown"
        confidence = 0.0
        action = action_for_label(label)
        bbox = "none"
        raw_bbox = "none"
        bbox_height = "none"
        class_name = "none"
        color_confidence = 0.0
        yolo_label = "unknown"
        crop_label = "unknown"
        lamp_position = "unknown"
        lamp_y_ratio = None
        raw_label = "unknown"
        decision_reason = "unknown"
        distance_m = None
        raw_distance_m = None
        bbox_distance_m = None
        zed_distance_m = None
        distance_source = "unknown"
        proximity = "unknown"
    else:
        label = detection["label"]
        confidence = detection["confidence"]
        action = detection["action"]
        bbox = detection["bbox"]
        raw_bbox = detection.get("raw_bbox", "none")
        bbox_height = detection.get("bbox_height", "unknown")
        class_name = detection["class_name"]
        color_confidence = detection["color_confidence"]
        yolo_label = detection.get("yolo_label", "unknown")
        crop_label = detection.get("crop_label", "unknown")
        lamp_position = detection.get("lamp_position", "unknown")
        lamp_y_ratio = detection.get("lamp_y_ratio")
        raw_label = detection.get("raw_label", label)
        decision_reason = detection.get("decision_reason", "unknown")
        distance_m = detection.get("distance_m")
        raw_distance_m = detection.get("raw_distance_m")
        bbox_distance_m = detection.get("bbox_distance_m")
        zed_distance_m = detection.get("zed_distance_m")
        distance_source = detection.get("distance_source", "unknown")
        proximity = detection.get("proximity", "unknown")

    current_state = f"{label}:{proximity}:{distance_m}"

    if current_state == previous_state:
        return previous_state

    print("\n================ YOLO TRAFFIC LIGHT LIVE ================")
    print(f"state: {label}")
    print(f"action: {action}")
    print(f"yolo_class: {class_name}")
    print(f"yolo_label: {yolo_label}")
    print(f"crop_label: {crop_label}")
    print(f"lamp_position: {lamp_position}")
    if lamp_y_ratio is None:
        print("lamp_y_ratio: unknown")
    else:
        print(f"lamp_y_ratio: {lamp_y_ratio:.3f}")
    print(f"raw_label: {raw_label}")
    print(f"decision_reason: {decision_reason}")
    print(f"yolo_confidence: {confidence:.2f}")
    print(f"color_confidence: {color_confidence:.3f}")

    if raw_distance_m is None:
        print("raw_distance: unknown")
    else:
        print(f"raw_distance: {raw_distance_m:.2f} m")

    if bbox_distance_m is None:
        print("bbox_distance: unknown")
    else:
        print(f"bbox_distance: {bbox_distance_m:.2f} m")

    if zed_distance_m is None:
        print("zed_distance: unknown")
    else:
        print(f"zed_distance: {zed_distance_m:.2f} m")

    if distance_m is None:
        print("distance: unknown")
    else:
        print(f"distance: {distance_m:.2f} m")

    print(f"distance_source: {distance_source}")
    print(f"proximity: {proximity}")
    print(f"bbox: {bbox}")
    print(f"raw_bbox_for_zed: {raw_bbox}")
    print(f"bbox_height: {bbox_height}")
    print("=========================================================\n")

    return current_state


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test traffic light detection using ZED camera."
    )

    parser.add_argument(
        "--detector",
        choices=("yolo", "keras"),
        default="yolo",
        help="Detector backend. Default uses YOLO live detection.",
    )

    parser.add_argument(
        "--yolo-model",
        default=DEFAULT_YOLO_MODEL_PATH,
        help="YOLO model path. Default uses src/model/best.pt.",
    )

    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.08,
        help="Minimum YOLO confidence. Low default helps detect distant traffic lights.",
    )

    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=1280,
        help="YOLO inference image size. Use 640 for speed, 960/1280 for distant traffic lights.",
    )

    parser.add_argument(
        "--yolo-valid-classes",
        default=(
            "traffic light,traffic_light,"
            "red,yellow,green,"
            "red_light,yellow_light,green_light,"
            "red_traffic_light,yellow_traffic_light,green_traffic_light"
        ),
        help="Comma-separated YOLO class names accepted as traffic lights.",
    )

    parser.add_argument(
        "--color-ratio-threshold",
        type=float,
        default=0.015,
        help="Minimum color ratio inside YOLO bbox when YOLO class is only traffic light.",
    )

    parser.add_argument(
        "--near-distance",
        type=float,
        default=0.6,
        help="Distance in meters below which the traffic light is considered close.",
    )

    parser.add_argument(
        "--far-distance",
        type=float,
        default=1.2,
        help="Distance in meters above which the traffic light is considered far.",
    )

    parser.add_argument(
        "--proximity-method",
        choices=("bbox_height", "distance"),
        default="bbox_height",
        help="How to classify close/medium/far. bbox_height is more stable for this test.",
    )

    parser.add_argument(
        "--near-bbox-height",
        type=int,
        default=130,
        help="BBox height in pixels above which the traffic light is considered close.",
    )

    parser.add_argument(
        "--far-bbox-height",
        type=int,
        default=85,
        help="BBox height in pixels below which the traffic light is considered far.",
    )

    parser.add_argument(
        "--model-path",
        default="",
        help="Full path to the .h5 model.",
    )

    parser.add_argument(
        "--image",
        default="",
        help="Path to a static image instead of ZED camera.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Minimum confidence required for a valid Keras class.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Keras model input width.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Keras model input height.",
    )

    parser.add_argument(
        "--roi-scale",
        type=float,
        default=1.0,
        help="Center crop size for Keras mode. Default 1.0 uses full frame.",
    )

    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Use full frame instead of center crop in Keras mode.",
    )

    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Manual crop area for Keras mode: x y width height.",
    )
    parser.add_argument(
        "--distance-method",
        choices=("bbox", "zed", "hybrid"),
        default="zed",
        help="Distance method. Default uses the ZED depth map, like sign detection.",
    )

    parser.add_argument(
        "--bbox-distance-k",
        type=float,
        default=182.0,
        help="Scale for bbox distance: distance = scale / bbox_height_px + offset.",
    )

    parser.add_argument(
        "--bbox-distance-offset",
        type=float,
        default=-0.89,
        help="Offset for bbox distance: distance = scale / bbox_height_px + offset.",
    )
    return parser.parse_args()


def open_zed_camera():
    if sl is None:
        raise RuntimeError(
            "ZED SDK Python API is not installed. Install pyzed first."
        )

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    # Avem nevoie de depth pentru distanța reală ZED.
    # Nu rula cu CUDA_VISIBLE_DEVICES="", pentru că ZED are nevoie de GPU.
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = ZED_DEPTH_MIN_M
    init_params.depth_maximum_distance = ZED_DEPTH_MAX_M

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED camera: {status}")

    return zed

def grab_zed_frame(zed, runtime_params, zed_image, zed_depth):
    status = zed.grab(runtime_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not grab frame from ZED: {status}")

    zed.retrieve_image(zed_image, sl.VIEW.LEFT)
    zed.retrieve_measure(zed_depth, sl.MEASURE.DEPTH)

    frame_bgra = zed_image.get_data()
    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

    # get_data() intoarce un view NumPy peste memoria detinuta de sl.Mat.
    # Bufferele sl.Mat sunt create o singura data in main() si raman vii cat
    # timp folosim aceste view-uri, evitand un use-after-free / segfault.
    depth_frame = zed_depth.get_data()

    return frame_bgr, depth_frame


def load_image(image_path):
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if frame is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return frame


def scan_frame(model, frame, args):
    input_size = (args.width, args.height)

    roi_frame, roi_box = get_roi(frame, args)

    if roi_frame.size == 0:
        raise RuntimeError("ROI crop is empty. Check --roi values.")

    label, confidence, probabilities, class_index = predict_label(
        model,
        roi_frame,
        args.threshold,
        input_size,
    )

    print_prediction(label, confidence, probabilities, class_index)

    cv2.imwrite("zed_scan_full_frame.jpg", frame)
    cv2.imwrite("zed_scan_roi_used_by_model.jpg", roi_frame)

    print("Saved full frame: zed_scan_full_frame.jpg")
    print("Saved ROI used by model: zed_scan_roi_used_by_model.jpg")

    result_frame = frame.copy()
    draw_result(result_frame, roi_box, label, confidence, probabilities)

    return result_frame


def load_detector(args):
    if args.detector == "yolo":
        if YOLO is None:
            raise RuntimeError(
                "ultralytics is not installed. Install it with:\n"
                "python3 -m pip install ultralytics"
            )

        yolo_model_path = resolve_yolo_model_path(args.yolo_model)

        print(f"Using YOLO model: {yolo_model_path}")
        print(f"YOLO confidence: {args.yolo_conf}")
        print(f"YOLO image size: {args.yolo_imgsz}")
        print(f"Near distance: {args.near_distance} m")
        print(f"Far distance: {args.far_distance} m")
        print("YOLO live mode: no S key needed. Press ESC to quit.")

        return YOLO(yolo_model_path)

    if keras is None:
        raise RuntimeError(
            "TensorFlow/Keras is not installed. Install it only if you want Keras mode."
        )

    if hub is None:
        raise RuntimeError(
            "tensorflow_hub is not installed. Install it only if you want Keras mode."
        )

    model_path = resolve_model_path(args.model_path)

    print(f"Using Keras model: {model_path}")
    print(f"Labels order: {LABELS}")
    print("IMPORTANT: Labels order must match train_ds.class_names from Colab.")

    return keras.models.load_model(
        model_path,
        custom_objects={"KerasLayer": hub.KerasLayer},
    )


def main():
    args = parse_args()
    model = load_detector(args)

    # Static image mode.
    if args.image:
        print(f"Using static image: {args.image}")

        frame = load_image(args.image)

        if args.detector == "yolo":
            detection = detect_yolo_traffic_light(
                model,
                frame,
                args,
                depth_frame=None,
            )
            print_yolo_detection(detection, previous_state=None)
            result_frame = draw_yolo_result(frame, detection)
        else:
            result_frame = scan_frame(model, frame, args)

        cv2.imshow("Traffic Light Image Scan Result", result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ZED camera mode.
    print("Opening ZED camera...")
    zed = open_zed_camera()
    print("ZED camera opened.")

    try:
        previous_state = None
        last_yolo_detection = None
        missed_detection_frames = 0
        runtime_params = sl.RuntimeParameters()
        zed_image = sl.Mat()
        zed_depth = sl.Mat()

        while True:
            frame, depth_frame = grab_zed_frame(
                zed,
                runtime_params,
                zed_image,
                zed_depth,
            )

            if args.detector == "yolo":
                detection = detect_yolo_traffic_light(
                    model,
                    frame,
                    args,
                    depth_frame=depth_frame,
                )   

                if detection is None:
                    missed_detection_frames += 1

                    if (
                        last_yolo_detection is not None
                        and missed_detection_frames <= MAX_MISSED_DETECTION_FRAMES
                    ):
                        detection = last_yolo_detection.copy()
                        detection["class_name"] = "last_valid_detection"
                    elif missed_detection_frames > MAX_MISSED_DETECTION_FRAMES:
                        reset_distance_filter()
                        reset_color_filter()
                        last_yolo_detection = None
                else:
                    detection = stabilize_detection_color(detection)
                    last_yolo_detection = detection.copy()
                    missed_detection_frames = 0

                previous_state = print_yolo_detection(detection, previous_state)
                display_frame = draw_yolo_result(frame, detection)
                window_name = "Traffic Light YOLO Live"
            else:
                result_frame = scan_frame(model, frame, args)
                display_frame = result_frame
                window_name = "Traffic Light Keras Live"

            cv2.imshow(window_name, display_frame)

            if cv2.waitKey(1) == 27:
                break

    finally:
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
