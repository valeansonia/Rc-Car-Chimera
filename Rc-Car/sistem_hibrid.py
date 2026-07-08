"""
sistem_hibrid.py  (v3.2 - detectie la distanta + eticheta "blocata" + corectie OCR viteza)
------------------
Sistem hibrid de detectie si clasificare a semnelor de circulatie:

    Camera ZED 2 --> YOLOv8 (best.pt)  --> gaseste UNDE e semnul (bbox)
                  --> MobileNetV2 (.h5) --> spune CE semn e (clasa GTSRB, 43 clase, RO)
                  --> OCR cifre viteza  --> corecteaza limitele care NU exista in GTSRB (90/110/130)
                  --> SignTracker        --> tine minte semnul intre cadre, ca sa nu
                                              mai "palpaie" eticheta

CE S-A SCHIMBAT FATA DE v3.1 (v3.2 -- fix pt confuzia 90->80 / 130->120):

    PROBLEMA GASITA: modelul de clasificare (model_semne_v2.h5) e antrenat
    pe GTSRB, care are DOAR 8 valori de viteza: 20/30/50/60/70/80/100/120.
    Limitele folosite frecvent la noi -- 90 (extravilan), 110, 130
    (autostrada) -- NU EXISTA ca si clase in acest model. Cand camera vede
    un semn real de 90 sau 130 km/h, CNN-ul e OBLIGAT sa aleaga cea mai
    apropiata clasa pe care o cunoaste -- de-aia apareau constant
    90 -> 80 si 130 -> 120 (nu era o problema de rezolutie sau de tracker,
    ci de vocabularul modelului).

    FIX: pt orice semn rotund + rosu (sau orice detectie deja clasificata
    ca limita de viteza), citim direct cifrele din crop prin segmentare de
    contur + template matching (functia read_speed_limit_number() din
    sign_detector.py, fara nicio dependinta noua, doar OpenCV):
      - daca OCR-ul confirma o valoare care EXISTA in GTSRB, o folosim ca
        sa "corectam" clasa CNN-ului acolo unde OCR-ul e mai increzator;
      - daca OCR-ul citeste o valoare care NU exista in GTSRB (90/110/130),
        afisam direct acea valoare (eticheta "Limita XX km/h"), ocolind
        limitarea de 43 de clase a modelului;
      - daca segmentarea cifrelor nu iese curat (semn prea mic/neclar/
        blurat), functia intoarce "nu stiu" si ramanem la ce a zis CNN-ul
        ca inainte -- deci acest fix nu poate strica ce mergea deja, poate
        doar corecta cazurile 90/110/130 (si, ca bonus, 60 vs 80 cand OCR-ul
        e suficient de increzator).

    Restul (detectie la distanta, scanare pe tile-uri, "blocarea" etichetei
    dupa recunoastere) e neschimbat fata de v3.1 -- vezi comentariile de mai
    jos, la fiecare sectiune.

Cerinte (neschimbate, + eventual h5py pt debugging, nu e nevoie in productie):
    pip install ultralytics

Fisiere necesare in acelasi folder:
    - sistem_hibrid.py          (acesta)
    - sign_detector.py          (imbunatatit, cu OCR cifre viteza)
    - gtsrb_labels.py           (imbunatatit, cu label_for_class())
    - model_semne_v2.h5         (modelul de clasificare)
    - best.pt / yolov8_semne.pt (modelul YOLO de detectie -- pune numele EXACT mai jos)
"""

import os
# NU mai ascundem GPU-ul la nivel de proces (CUDA_VISIBLE_DEVICES) -- placa
# video trebuie sa ramana vizibila pt SDK-ul ZED 2, care foloseste CUDA ca sa
# calculeze harta de adancime (distanta pana la semn). TensorFlow e in
# continuare fortat pe CPU, dar DOAR pt el, in sign_detector.py (vezi
# tf.config.set_visible_devices([], "GPU") de acolo) -- asa ZED-ul isi
# pastreaza accesul la GPU si adancimea chiar merge.

import sys
sys.path.append("/home/wsadmin/.local/lib/python3.12/site-packages")

import time
from collections import deque, Counter

import cv2
import numpy as np
import pyzed.sl as sl                       # noqa: F401  (path setup)

from ultralytics import YOLO

from sign_detector import (
    TrafficSignClassifier,
    _classify_shape,
    _color_masks,
    shape_color_to_category,
    classify_speed_limit,
)
from gtsrb_labels import (
    GTSRB_LABELS_RO,
    SHAPE_TO_LIKELY_CLASSES,
    SPEED_LIMIT_CLASSES,
    label_for_class,
    likely_classes_for_category,
)


# =====================================================================
# CONFIG
# =====================================================================
YOLO_MODEL_PATH = "yolov8_semne.pt"    # ATENTIE: pune aici numele EXACT al fisierului tau .pt
CLASSIFIER_MODEL_PATH = "model_semne_v2.h5"

# =====================================================================
# IMPORTANT: yolov8_semne.pt NU e un simplu "localizator de semne" generic
# -- e deja antrenat cu 50 de clase PROPRII (localizare + clasificare
# intr-un singur model), incluzand exact tipurile de semne care nu mergeau
# deloc pana acum: "Yield", "Traffic Signals", "priority road", curbe,
# sens interzis etc. (verificat direct din model.names). Codul vechi
# arunca complet aceasta informatie -- lua doar bbox-ul si incredere de la
# YOLO, apoi reclasifica de la zero prin CNN-ul mic (model_semne_v2.h5) +
# euristica de forma/culoare, care s-a dovedit fragila pt forme cu chenar
# subtire (triunghi/romb). Testat direct: YOLO singur clasifica CORECT un
# triunghi "Traffic Signals" simulat, cu incredere rezonabila.
#
# Solutie: pt semnele care NU sunt limita de viteza, folosim DIRECT clasa
# proprie a lui YOLO (mult mai fiabil pt pictograme decat euristica), fara
# sa mai trecem deloc prin CNN-ul mic / shape-color/OCR -- mai rapid SI mai
# corect. Pt limitele de viteza, pipeline-ul rama NESCHIMBAT (CNN + OCR),
# pt ca deja functioneaza foarte bine si nu vrem sa-l stricam.
#
# Mapare index YOLO (model.names) -> class_id GTSRB folosit de restul
# sistemului (gtsrb_labels.GTSRB_LABELS_RO). Indecsii 0/1/2 din model
# ("20"/"21"/"22") sunt ambigui/nefolositi (resturi din antrenare, nu
# corespund unor clase GTSRB valide) -- ii lasam nemapati intentionat, ca
# sa cada pe calea veche (CNN + euristica) daca YOLO ii prezice vreodata.
YOLO_CLASS_TO_GTSRB = {
    3: 35, 4: 30, 5: 22, 6: 28, 7: 19, 8: 20, 9: 21, 10: 32, 11: 42, 12: 41,
    13: 18, 14: 37, 15: 36, 16: 39, 17: 38, 18: 17, 19: 9, 20: 10, 21: 15,
    22: 27, 23: 11, 24: 24, 25: 25, 26: 40, 27: 23, 28: 26, 29: 34, 30: 33,
    31: 31, 32: 13, 33: 35, 34: 29, 35: 28, 36: 6, 37: 12, 38: 40,
    39: 7, 40: 8, 41: 0, 42: 1, 43: 2, 44: 3, 45: 4, 46: 5,
    47: 14, 48: 34, 49: 16,
}
# Indecsii YOLO care sunt EI INSISI limite de viteza -- pt astea pastram
# calea veche (CNN + OCR), NU calea rapida de mai sus, ca sa nu stricam
# nimic din ce am reparat deja pt 20/30/50/60/70/80/90/100/110/120/130.
SPEED_YOLO_CLASS_IDS = {39, 40, 41, 42, 43, 44, 45, 46}
# Cat de increzator trebuie sa fie YOLO in PROPRIA lui clasa ca sa o
# folosim direct (fara sa mai trecem prin CNN-ul mic). Daca YOLO nu e
# suficient de sigur, cadem pe calea veche (safety net), nu pierdem nimic.
YOLO_CLASS_CONF_THRESHOLD = 0.35

# --- praguri de detectie / clasificare ---
YOLO_CONF_THRESHOLD = 0.25          # cobort de la 0.35 -> prinde si semnele mici/departate
CLASSIFIER_CONF_THRESHOLD = 0.45    # cobort de la 0.60 -> decizia finala vine din tracker (vot ponderat), nu dintr-un cadru
BOX_PADDING_RATIO = 0.12
BOX_PADDING_RATIO_SMALL = 0.25      # padding mai mare pt semne mici (mai mult context pt clasificator)
SMALL_BOX_SIZE_PX = 45              # sub cati pixeli (latura mica a bbox-ului) consideram semnul "de departe"
USE_SHAPE_COLOR_PRIOR = True
SHAPE_COLOR_PENALTY = 0.7           # forma+culoarea sunt valide, dar pt alta categorie de semn decat a zis CNN-ul
SHAPE_COLOR_NO_MATCH_PENALTY = 0.15 # forma+culoarea NU corespund niciunui semn real (perete, haine, etc.)
SHAPE_COLOR_UNKNOWN_PENALTY = 0.35  # nu am reusit sa extragem deloc o forma/culoare clara din decupaj
MIN_COLOR_COVERAGE_RATIO = 0.25     # cat % din decupaj trebuie sa fie efectiv rosu/albastru/galben (nu doar nuanta medie) -- pt forme cu CHENAR GROS (cerc/octogon)
# Triunghiurile de avertizare (cedeaza trecerea, semafoare, pietoni etc.) si
# rombul galben (drum cu prioritate) au un chenar SUBTIRE pe un interior ALB
# -- culoarea reala acopera mult sub 25% din cutie, spre deosebire de cercul
# de la limitele de viteza (inel rosu gros). Cu un singur prag de 0.25,
# aproape orice semn triunghiular/romb era respins de acest filtru, oricat
# de bine il clasifica CNN-ul -- de-aia nu se detectau deloc. Prag separat,
# mult mai permisiv, pt formele astea.
MIN_COLOR_COVERAGE_RATIO_THIN_BORDER = 0.08
MIN_BOX_SIZE_FOR_SHAPE_PRIOR = 40
# Pragurile astea doua erau tinute prea stranse (0.6-1.6 aspect, 30% arie),
# calibrate initial doar impotriva unui perete/hartie goala care umplea
# cadrul -- dar acelasi filtru respingea si semne REALE tinute aproape de
# camera (cutia depaseste usor 30% din cadru la distanta mica) sau vazute
# putin oblic/lateral (perspectiva schimba aspectul bbox-ului fata de un
# patrat perfect). Acum ca poarta forma+culoare e principala aparare
# impotriva fals-pozitivelor, astea doua raman doar un plafon de bun-simt
# impotriva cazurilor absurde (cutie aproape cat tot cadrul, sau extrem de
# alungita), nu mai trebuie sa faca toata munca singure.
MIN_BOX_ASPECT_RATIO = 0.45         # semnele reale (cerc/triunghi/octogon/romb) sunt aprox. patrate in bbox, dar tolerant la unghi
MAX_BOX_ASPECT_RATIO = 2.2          # -- resping doar cutii FOARTE alungite (muchii de perete, haine, etc.)
MAX_BOX_AREA_RATIO = 0.60           # un semn tinut aproape de camera poate umple usor jumatate din cadru -- plafon doar impotriva cazurilor absurde
# YOLO_IMGSZ = rezolutia la care ruleaza trecerea de baza, PE FIECARE cadru.
# Scazut de la 1280 -> 960 ca sa castigam viteza (costul YOLO pe CPU creste
# aprox. patratic cu rezolutia, deci 960 e considerabil mai rapid ca 1280).
# NU afecteaza calitatea detectiei la distanta: semnele mici/departate tot
# sunt prinse de scanarea pe tile-uri (FAR_SCAN_IMGSZ ramane 1280, neschimbat
# mai jos), care ruleaza separat, periodic, exact pt cazul asta.
YOLO_IMGSZ = 960                     # inainte: 1280. Poti urca inapoi la 1280 daca ai nevoie de mai multa precizie si nu de viteza.

# --- corectie OCR pt limitele de viteza (vezi explicatia de la inceputul fisierului) ---
ENABLE_SPEED_OCR_OVERRIDE = True     # pune False daca vrei sa revii la comportamentul dinainte (fara OCR)
SPEED_OCR_MIN_DIGIT_CONF = 0.50      # cat de sigur trebuie sa fie template matching-ul pe FIECARE cifra (usor coborat de la 0.55, acum ca avem si mai multe fonturi sablon)
# Cand OCR-ul chiar a citit cifrele (nu doar a "castigat" fata de CNN), acea
# citire conteaza ca dovada mult mai directa decat un simplu vot CNN --
# mai ales pt 90/110/130, pe care CNN-ul nu le poate spune NICIODATA din
# constructie. O ponderam mai mult in istoricul tracker-ului (inmultind
# increderea trimisa la vot), ca eticheta corecta sa castige rapid votul
# ponderat chiar daca au fost deja cateva cadre gresite de la CNN inainte
# sa prindem un citit OCR clar -- asta e principalul motiv pt care semnele
# 90/110/130 pareau ca "dureaza mult" sa fie recunoscute corect.
OCR_VOTE_BOOST = 1.6

FRAME_FPS = 30

# ---- adancime (distanta pana la semn) prin senzorul stereo al ZED 2 -------
# IMPORTANT: camera nu mai e deschisa prin cv2.VideoCapture(index) (acces
# UVC brut, care doar citea imaginea stereo alaturata fara nicio calibrare/
# rectificare) -- e deschisa prin SDK-ul ZED (pyzed.sl), singurul mod prin
# care putem cere si harta de adancime (Z) calculata de camera din
# perechea stereo. SDK-ul gestioneaza el insusi rezolutia/FPS-ul camerei
# fizice; ZED_RESOLUTION de mai jos e rezolutia PE OCHI (nu latimea totala
# a perechii stereo, cum era FRAME_WIDTH inainte).
ZED_RESOLUTION = sl.RESOLUTION.HD720        # 1280x720 pe ochi -- acelasi cadru ca inainte (2560/2 x 720)
# PERFORMANCE dadea citiri de adancime vizibil instabile (sarea de la un
# cadru la altul chiar si pt un semn tinut nemiscat) -- QUALITY e un pas
# binecunoscut de precizie in SDK-ul ZED, cu un cost de viteza moderat;
# calculul de adancime ruleaza pe GPU, separat de bucla CPU de YOLO/CNN, deci
# nu ar trebui sa incetineasca deloc partea de detectie/clasificare.
ZED_DEPTH_MODE = sl.DEPTH_MODE.QUALITY
ZED_DEPTH_MIN_M = 0.3                       # sub atat, adancimea ZED e oricum nesigura (prea aproape de senzor)
ZED_DEPTH_MAX_M = 20.0                      # peste atat, nu ne intereseaza (semnul oricum abia se distinge)
DISTANCE_BOX_MARGIN_RATIO = 0.25            # ignoram marginea cutiei la esantionarea adancimii (fundal/contur semn)
# O SINGURA citire de adancime poate sari mult intre cadre (zgomot de
# stereo matching, gauri in harta de adancime) -- pastram un mic istoric de
# citiri BRUTE recente per track si folosim MEDIANA lor (robusta la un
# outlier izolat) inainte de netezirea exponentiala finala. Alpha-ul de
# netezire a fost si el coborat (0.25 -> 0.15) pt un rezultat mult mai
# stabil cand semnul chiar nu se misca -- raspunde putin mai incet la
# schimbari reale de distanta, dar nu mai "tremura" pe loc.
DISTANCE_HISTORY_LEN = 5
DISTANCE_SMOOTHING_ALPHA = 0.15             # netezire intre cadre (media exponentiala), dupa filtrarea prin mediana

# ---- setari pentru tracker-ul anti-flicker ----
TRACK_IOU_MATCH_THRESHOLD = 0.30    # usor mai permisiv (semnele mici se misca mai mult relativ la marimea lor)
TRACK_HISTORY_LEN = 15              # cate predictii recente tine minte fiecare semn urmarit
MIN_VOTE_RATIO = 0.55               # (folosit cat timp track-ul NU e inca "blocat")
TRACK_MAX_MISSED_FRAMES = 10
# Numarul MINIM de cicluri (cadre) in care semnul trebuie sa fi fost vazut
# la rand inainte sa dam vreun verdict (sa desenam eticheta). Le tinusem
# ridicate (5/10) cat timp verificarea forma+culoare avea o gaura (nuanta
# medie, fara saturatie) care lasa pereti/haine sa treaca drept semne --
# atunci singura aparare era sa cerem multe cadre la rand. Acum ca filtrele
# PE CADRU sunt solide (masca HSV cu prag de acoperire, plafon de arie,
# aspect ratio -- vezi mai jos), nu mai avem nevoie de atata persistenta
# temporala doar ca sa respingem fals-pozitive, asa ca le-am scazut la loc,
# ca sa dam un verdict mult mai repede pt semnele reale.
CONFIRM_FRAMES_NEAR = 2             # cadre de confirmare pt semne aproape/mari (detectie sigura)
CONFIRM_FRAMES_FAR = 4              # cadre de confirmare pt semne mici/departate (mai predispuse la fals-pozitiv)
BOX_SMOOTHING_ALPHA = 0.30

# ---- "blocarea" etichetei dupa recunoastere clara (anti-schimbare de "focus") ----
LOCK_MIN_FRAMES = 4         # dupa cate cadre de vot clar consideram eticheta stabila (redus de la 8, vezi motivul de mai sus)
LOCK_MIN_RATIO = 0.70       # ponderea (din increderi) pe care clasa dominanta trebuie sa o aiba ca sa se blocheze
LOCK_MIN_AVG_CONF = 0.60    # increderea medie minima a clasei dominante ca sa se blocheze
UNLOCK_MIN_CONF = 0.75      # o clasa noua trebuie sa apara cu incredere mare ca sa "sparga" blocajul...
UNLOCK_CONSECUTIVE = 5      # ...CONSECUTIV, atatea cadre la rand...
UNLOCK_MIN_RATIO = 0.80     # ...si sa domine clar istoricul recent

# ---- scanare pe TILE-uri pt semne FOARTE mici / la distanta mare ----
# NESCHIMBATA (frecventa + rezolutie) -- e mecanismul care garanteaza
# detectia semnelor mici/departate, nu vrem sa-i scadem calitatea sau sa-l
# rulam mai rar doar ca sa castigam viteza. Viteza am castigat-o din
# YOLO_IMGSZ (mai sus), care nu afecteaza deloc scanarea asta.
ENABLE_FAR_SCAN = True       # pune False daca vrei sa dezactivezi (revii la comportamentul dinainte)
FAR_SCAN_EVERY_N_FRAMES = 3  # scaneaza pe tile-uri o data la atatea cadre
TILE_GRID = (2, 2)           # (randuri, coloane) -- 2x2 = imaginea taiata in 4 bucati care se suprapun putin
TILE_OVERLAP_RATIO = 0.15    # suprapunere intre tile-uri vecine, ca un semn de la marginea unui tile sa nu fie ratat
FAR_YOLO_CONF_THRESHOLD = 0.15  # prag de incredere YOLO, mai permisiv DOAR pt scanarea pe tile-uri
FAR_SCAN_IMGSZ = 1280        # rezolutia la care e marit FIECARE tile inainte sa intre in YOLO
MERGE_IOU_THRESHOLD = 0.55   # cat de mult trebuie sa se suprapuna doua bbox-uri ca sa fie combinate (evita chenare duble)
# NOTA: marit usor de la 0.45 -- daca pui MAI MULTE semne alaturate/apropiate
# in fata camerei, cutiile lor pot sa se atinga/suprapuna partial fara sa
# fie de fapt acelasi semn; un prag prea mic le-ar combina gresit intr-una
# singura, pierzand semnele individuale din grup.

# semnul ramane desenat pe ecran atatea cadre dupa ULTIMA detectie reala,
# ca sa nu "clipeasca" intre doua scanari pe tile-uri
DISPLAY_GRACE_FRAMES = FAR_SCAN_EVERY_N_FRAMES


# =====================================================================
# ADANCIME (distanta pana la semn) -- citita din harta de adancime a ZED 2
# =====================================================================
def estimate_distance_m(depth_map, bbox):
    """Citeste distanta (in metri) pana la semnul din `bbox`, folosind harta
    de adancime calculata de ZED 2 din perechea stereo.

    De ce mediana pe o zona centrala si nu un singur pixel din centrul
    cutiei: harta de adancime a ZED are frecvent "gauri" (NaN/inf) exact pe
    marginile/conturul obiectelor (acolo unde stereo matching-ul esueaza),
    iar un singur pixel putea pica exact intr-o astfel de gaura sau pe
    fundal (daca bbox-ul YOLO e putin mai mare decat semnul real). Folosim
    mediana valorilor valide dintr-o zona centrala (fara marginea cutiei),
    ceea ce da o citire stabila chiar daca o parte din zona e invalida.

    Returneaza distanta in metri (float) sau None daca nu avem harta de
    adancime sau nu exista nicio citire valida in zona (semn prea aproape/
    departe de limitele senzorului, sau zona complet in "gaura" de adancime).
    """
    if depth_map is None:
        return None

    x, y, w, h = bbox
    mx, my = int(w * DISTANCE_BOX_MARGIN_RATIO), int(h * DISTANCE_BOX_MARGIN_RATIO)
    x0, y0 = max(0, x + mx), max(0, y + my)
    x1, y1 = min(depth_map.shape[1], x + w - mx), min(depth_map.shape[0], y + h - my)
    if x1 <= x0 or y1 <= y0:
        return None

    region = depth_map[y0:y1, x0:x1]
    valid = region[np.isfinite(region) & (region >= ZED_DEPTH_MIN_M) & (region <= ZED_DEPTH_MAX_M)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


# =====================================================================
# TRACKER ANTI-FLICKER
# =====================================================================
def _iou(box_a, box_b):
    """Intersection-over-Union intre doua bbox-uri (x, y, w, h)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class _Track:
    """Un singur semn urmarit de-a lungul mai multor cadre."""
    _next_id = 1

    def __init__(self, detection):
        self.id = _Track._next_id
        _Track._next_id += 1

        self.bbox = detection["bbox"]                       # bbox netezit (afisat)
        self.distance_m = detection.get("distance_m")        # distanta netezita (afisata)
        # istoric scurt de citiri BRUTE de adancime -- vezi update(), unde
        # folosim mediana lor (robusta la un outlier izolat) inainte de
        # netezirea exponentiala
        self.distance_history = deque(maxlen=DISTANCE_HISTORY_LEN)
        if self.distance_m is not None:
            self.distance_history.append(self.distance_m)
        # istoricul retine (class_id, confidence) -- confidence e folosita ca
        # pondere in vot, nu doar o numaratoare bruta
        self.class_history = deque(maxlen=TRACK_HISTORY_LEN)
        self.class_history.append((detection["class_id"], detection["confidence"]))

        self.stable_class_id = detection["class_id"]
        self.stable_label = detection["label_ro"]
        self.last_confidence = detection["confidence"]

        self.missed_frames = 0
        self.seen_frames = 1

        # cate cadre de confirmare sunt necesare inainte de a desena semnul
        # (mai multe pt semnele mici/departate, ca sa nu iasa fals-pozitive)
        self.confirm_frames_needed = detection.get("confirm_frames_needed", CONFIRM_FRAMES_NEAR)
        self.confirmed = self.confirm_frames_needed <= 1

        # stare de "blocare" a etichetei (anti-flicker intre clase asemanatoare,
        # ex. limitele de viteza 100 vs 120)
        self.locked = False
        self.locked_frames_at_top = 1
        self.challenger_class = None
        self.challenger_count = 0

    def _weighted_vote(self):
        """Calculeaza ponderea (suma increderilor) pt fiecare clasa din
        istoric si returneaza (clasa_dominanta, raport, incredere_medie_clasa)."""
        weights = {}
        conf_sums = {}
        counts = {}
        for cid, conf in self.class_history:
            weights[cid] = weights.get(cid, 0.0) + conf
            conf_sums[cid] = conf_sums.get(cid, 0.0) + conf
            counts[cid] = counts.get(cid, 0) + 1

        total = sum(weights.values())
        if total <= 0:
            return self.stable_class_id, 1.0, self.last_confidence

        top_class = max(weights, key=weights.get)
        ratio = weights[top_class] / total
        avg_conf = conf_sums[top_class] / counts[top_class]
        return top_class, ratio, avg_conf

    def _weighted_vote_recent(self, n):
        """La fel ca _weighted_vote(), dar calculat DOAR pe ultimele n
        predictii. Folosit la verificarea de "unlock"."""
        recent = list(self.class_history)[-n:]
        weights = {}
        for cid, conf in recent:
            weights[cid] = weights.get(cid, 0.0) + conf
        total = sum(weights.values())
        if total <= 0:
            return self.stable_class_id, 1.0
        top_class = max(weights, key=weights.get)
        ratio = weights[top_class] / total
        return top_class, ratio

    def update(self, detection):
        # netezeste bbox-ul (media ponderata intre pozitia veche si cea noua)
        ox, oy, ow, oh = self.bbox
        nx, ny, nw, nh = detection["bbox"]
        a = BOX_SMOOTHING_ALPHA
        self.bbox = (
            int(ox + a * (nx - ox)),
            int(oy + a * (ny - oy)),
            int(ow + a * (nw - ow)),
            int(oh + a * (nh - oh)),
        )

        # netezeste distanta la fel ca bbox-ul -- o singura citire de
        # adancime poate sari (gauri in harta de adancime, zgomot stereo la
        # distanta mare), asa ca o mediem exponential intre cadre in loc sa
        # afisam bruta ultima valoare citita
        new_distance = detection.get("distance_m")
        if new_distance is not None:
            self.distance_history.append(new_distance)
            # mediana ultimelor DISTANCE_HISTORY_LEN citiri BRUTE -- respinge
            # un singur cadru cu zgomot (gaura in harta de adancime, citire
            # aiurea) inainte sa ajunga la netezirea exponentiala de mai jos
            filtered_distance = float(np.median(self.distance_history))
            if self.distance_m is None:
                self.distance_m = filtered_distance
            else:
                b = DISTANCE_SMOOTHING_ALPHA
                self.distance_m = self.distance_m + b * (filtered_distance - self.distance_m)

        new_cls = detection["class_id"]
        new_conf = detection["confidence"]

        self.class_history.append((new_cls, new_conf))
        self.last_confidence = new_conf
        self.missed_frames = 0
        self.seen_frames += 1
        if not self.confirmed and self.seen_frames >= self.confirm_frames_needed:
            self.confirmed = True

        top_class, ratio, avg_conf = self._weighted_vote()

        if not self.locked:
            # comportament ca inainte (vot majoritar), dar ponderat cu increderea
            if ratio >= MIN_VOTE_RATIO:
                self.stable_class_id = top_class
                self.stable_label = label_for_class(top_class)

            # verificam daca eticheta a devenit suficient de stabila ca sa o "blocam"
            if top_class == self.stable_class_id:
                self.locked_frames_at_top += 1
            else:
                self.locked_frames_at_top = 1

            if (self.locked_frames_at_top >= LOCK_MIN_FRAMES
                    and ratio >= LOCK_MIN_RATIO
                    and avg_conf >= LOCK_MIN_AVG_CONF):
                self.locked = True
                self.challenger_class = None
                self.challenger_count = 0

        else:
            # eticheta e "blocata": o schimbam DOAR daca apare un provocator
            # clar, cu incredere mare, sustinut mai multe cadre la rand
            if new_cls != self.stable_class_id and new_conf >= UNLOCK_MIN_CONF:
                if self.challenger_class == new_cls:
                    self.challenger_count += 1
                else:
                    self.challenger_class = new_cls
                    self.challenger_count = 1
            else:
                self.challenger_class = None
                self.challenger_count = 0

            if (self.challenger_class is not None
                    and self.challenger_count >= UNLOCK_CONSECUTIVE):
                recent_top, recent_ratio = self._weighted_vote_recent(UNLOCK_CONSECUTIVE)
                if recent_top == self.challenger_class and recent_ratio >= UNLOCK_MIN_RATIO:
                    self.stable_class_id = self.challenger_class
                    self.stable_label = label_for_class(self.challenger_class)
                    self.locked = False
                    self.locked_frames_at_top = 1
                    self.challenger_class = None
                    self.challenger_count = 0

    def mark_missed(self):
        self.missed_frames += 1

    def is_dead(self):
        return self.missed_frames > TRACK_MAX_MISSED_FRAMES

    def as_display_dict(self):
        return {
            "bbox": self.bbox,
            "class_id": self.stable_class_id,
            "label_ro": self.stable_label,
            "confidence": self.last_confidence,
            "track_id": self.id,
            "locked": self.locked,
            "distance_m": self.distance_m,
        }


class SignTracker:
    """Potriveste detectiile brute (de la YOLO+clasificator) cu semne
    urmarite intre cadre, ca eticheta afisata sa nu mai sara de la un
    cadru la altul."""

    def __init__(self):
        self.tracks = []

    def update(self, detections):
        unmatched_tracks = list(self.tracks)
        unmatched_detections = list(detections)
        matches = []  # (track, detection)

        pairs = []
        for t in unmatched_tracks:
            for d in unmatched_detections:
                iou = _iou(t.bbox, d["bbox"])
                if iou >= TRACK_IOU_MATCH_THRESHOLD:
                    pairs.append((iou, t, d))
        pairs.sort(key=lambda p: -p[0])

        used_tracks, used_dets = set(), set()
        for iou, t, d in pairs:
            if id(t) in used_tracks or id(d) in used_dets:
                continue
            matches.append((t, d))
            used_tracks.add(id(t))
            used_dets.add(id(d))

        for t, d in matches:
            t.update(d)
            unmatched_tracks.remove(t)
            unmatched_detections.remove(d)

        for t in unmatched_tracks:
            t.mark_missed()

        for d in unmatched_detections:
            self.tracks.append(_Track(d))

        self.tracks = [t for t in self.tracks if not t.is_dead()]

        return [
            t.as_display_dict()
            for t in self.tracks
            if t.confirmed and t.missed_frames <= DISPLAY_GRACE_FRAMES
        ]


# =====================================================================
# INITIALIZARE MODELE
# =====================================================================
def load_models():
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Eroare: nu gasesc modelul YOLO '{YOLO_MODEL_PATH}' in acest folder!")
        sys.exit(1)
    if not os.path.exists(CLASSIFIER_MODEL_PATH):
        print(f"Eroare: nu gasesc modelul de clasificare '{CLASSIFIER_MODEL_PATH}' in acest folder!")
        sys.exit(1)

    print("Se incarca YOLO (detectie semne)...")
    detector = YOLO(YOLO_MODEL_PATH)
    detector.to("cpu")

    print("Se incarca MobileNetV2 (clasificare semne)... poate dura cateva secunde.")
    classifier = TrafficSignClassifier(CLASSIFIER_MODEL_PATH)

    print("Ambele modele au fost incarcate cu succes.")
    return detector, classifier


# =====================================================================
# DETECTIE PE TILE-URI (pt semne foarte mici / la distanta mare)
# =====================================================================
def _generate_tiles(w_img, h_img, grid=TILE_GRID, overlap_ratio=TILE_OVERLAP_RATIO):
    """Imparte imaginea in `grid` = (randuri, coloane) bucati care se
    suprapun putin intre ele."""
    rows, cols = grid
    tile_w = w_img / cols
    tile_h = h_img / rows
    overlap_w = tile_w * overlap_ratio
    overlap_h = tile_h * overlap_ratio

    tiles = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, int(c * tile_w - overlap_w))
            y0 = max(0, int(r * tile_h - overlap_h))
            x1 = min(w_img, int((c + 1) * tile_w + overlap_w))
            y1 = min(h_img, int((r + 1) * tile_h + overlap_h))
            tiles.append((x0, y0, x1, y1))
    return tiles


def _run_yolo_on_region(detector, frame_bgr, region, conf, imgsz):
    """Ruleaza YOLO doar pe o bucata (tile) din imagine, marita separat la
    `imgsz`."""
    x0, y0, x1, y1 = region
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return []

    results = detector.predict(
        crop, conf=conf, imgsz=imgsz, verbose=False, device="cpu",
    )[0]

    boxes = []
    if results.boxes is not None and len(results.boxes) > 0:
        xyxy = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)
        for (bx1, by1, bx2, by2), c, k in zip(xyxy, confs, clss):
            boxes.append((bx1 + x0, by1 + y0, bx2 + x0, by2 + y0, float(c), int(k)))
    return boxes


def _merge_boxes_nms(boxes, iou_thresh=MERGE_IOU_THRESHOLD):
    """Combina bbox-urile venite din trecerea normala (tot cadrul) cu cele
    din tile-uri, eliminand duplicatele."""
    boxes_sorted = sorted(boxes, key=lambda b: -b[4])
    kept = []
    for bx1, by1, bx2, by2, bconf, bcls in boxes_sorted:
        bw, bh = bx2 - bx1, by2 - by1
        is_dup = False
        for kx1, ky1, kx2, ky2, _, _ in kept:
            iou = _iou((bx1, by1, bw, bh), (kx1, ky1, kx2 - kx1, ky2 - ky1))
            if iou >= iou_thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append((bx1, by1, bx2, by2, bconf, bcls))
    return kept


def _detect_shape_color(roi_strict):
    """Calculeaza forma si culoarea dominante dintr-un decupaj STRICT (fara
    padding) al unui candidat de semn. Returneaza (shape_name, color_name)
    -- oricare poate fi None daca nu s-a putut determina clar (nu e neaparat
    un semn real, ex. perete/hartie alba/haine).

    ATENTIE: aceasta e verificarea de siguranta principala impotriva
    fals-pozitivelor (nu doar un "bonus" de acuratete) -- vezi cum e
    folosita mai jos, in ambele cai (rapida SI veche), din detect_and_
    classify_hybrid().
    """
    if roi_strict is None or roi_strict.size == 0:
        return None, None
    try:
        gray = cv2.cvtColor(roi_strict, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_name = None
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            shape_name = _classify_shape(biggest)

        hsv_roi = cv2.cvtColor(roi_strict, cv2.COLOR_BGR2HSV)
        color_masks_roi = _color_masks(hsv_roi)
        roi_pixel_count = roi_strict.shape[0] * roi_strict.shape[1]
        best_color, best_coverage = None, 0.0
        for cname, cmask in color_masks_roi.items():
            coverage = float(np.count_nonzero(cmask)) / roi_pixel_count
            if coverage > best_coverage:
                best_coverage, best_color = coverage, cname

        min_coverage_needed = (
            MIN_COLOR_COVERAGE_RATIO_THIN_BORDER
            if shape_name in ("triangle", "diamond")
            else MIN_COLOR_COVERAGE_RATIO
        )
        color_name = best_color if best_coverage >= min_coverage_needed else None
        return shape_name, color_name
    except Exception:
        return None, None


# =====================================================================
# PIPELINE: YOLO detecteaza -> crop -> MobileNetV2 clasifica -> OCR corecteaza viteza
# =====================================================================
def detect_and_classify_hybrid(frame_bgr, detector, classifier, frame_idx=0,
                                yolo_conf=YOLO_CONF_THRESHOLD,
                                cls_conf=CLASSIFIER_CONF_THRESHOLD,
                                use_shape_prior=USE_SHAPE_COLOR_PRIOR,
                                yolo_imgsz=YOLO_IMGSZ,
                                depth_map=None):
    h_img, w_img = frame_bgr.shape[:2]

    # --- 1) trecerea normala, pe tot cadrul ---
    yolo_results = detector.predict(
        frame_bgr,
        conf=yolo_conf,
        imgsz=yolo_imgsz,
        verbose=False,
        device="cpu",
    )[0]

    raw_boxes = []
    if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
        boxes_xyxy = yolo_results.boxes.xyxy.cpu().numpy()
        yolo_confs = yolo_results.boxes.conf.cpu().numpy()
        yolo_clss = yolo_results.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(boxes_xyxy, yolo_confs, yolo_clss):
            raw_boxes.append((float(x1), float(y1), float(x2), float(y2), float(c), int(k)))

    # --- 2) din cand in cand, o trecere suplimentara PE TILE-URI ---
    if ENABLE_FAR_SCAN and (frame_idx % FAR_SCAN_EVERY_N_FRAMES == 0):
        for region in _generate_tiles(w_img, h_img):
            raw_boxes.extend(
                _run_yolo_on_region(detector, frame_bgr, region,
                                     FAR_YOLO_CONF_THRESHOLD, FAR_SCAN_IMGSZ)
            )

    # IMPORTANT: filtram forma/aria PE CUTIILE BRUTE, INAINTE de combinarea
    # (dedup) intre trecerea de baza si tile-uri. Motiv: daca mai multe
    # semne sunt puse alaturat/apropiat (ex. toate semnele deodata in fata
    # camerei), YOLO poate desena din greseala O cutie mare peste tot
    # grupul. Daca am fi filtrat DUPA combinare, acea cutie mare (gresita)
    # ar fi "castigat" dedup-ul fata de cutiile mici si corecte ale
    # semnelor individuale (IOU mare intre ele) -- pierzand toate semnele
    # din grup, chiar daca filtrul de arie oricum ar fi respins-o imediat
    # dupa. Filtrand INAINTE, cutia mare gresita e eliminata devreme si nu
    # mai apuca sa suprime alternativele bune la combinare.
    def _is_plausible_sign_box(x1, y1, x2, y2):
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return False
        aspect_ratio = w / float(h)
        if aspect_ratio < MIN_BOX_ASPECT_RATIO or aspect_ratio > MAX_BOX_ASPECT_RATIO:
            return False
        if (w * h) > MAX_BOX_AREA_RATIO * (w_img * h_img):
            return False
        return True

    raw_boxes = [b for b in raw_boxes if _is_plausible_sign_box(b[0], b[1], b[2], b[3])]

    merged_boxes = _merge_boxes_nms(raw_boxes)

    detections = []
    if not merged_boxes:
        return detections

    for (x1, y1, x2, y2, yolo_conf_score, yolo_cls_id) in merged_boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue

        is_small = min(w, h) < SMALL_BOX_SIZE_PX
        roi_strict = frame_bgr[y1:y2, x1:x2]
        box_big_enough = min(w, h) >= MIN_BOX_SIZE_FOR_SHAPE_PRIOR

        # Calculam forma+culoarea O SINGURA DATA (folosita mai jos si de
        # calea rapida, si de calea veche) -- vezi _detect_shape_color().
        shape_name, color_name = (None, None)
        if use_shape_prior and box_big_enough:
            shape_name, color_name = _detect_shape_color(roi_strict)

        # --- Calea RAPIDA: semne care NU sunt limita de viteza -------------
        # YOLO stie deja singur ce clasa e (a fost antrenat cu 50 de clase
        # proprii) -- pt pictograme (cedeaza trecerea, semafoare, drum cu
        # prioritate, curbe etc.) e mult mai fiabil decat sa reclasificam de
        # la zero cu CNN-ul mic. O folosim direct, dar NUMAI daca forma+
        # culoarea decupajului CONFIRMA independent categoria prezisa de
        # YOLO (ex. YOLO zice "Cedeaza trecerea" -> trebuie sa vedem chiar
        # noi un triunghi rosu in decupaj).
        #
        # ATENTIE, aceasta verificare NU e optionala -- fara ea, doua lucruri
        # se strica: (1) YOLO poate "vota" cu incredere peste pragul minim
        # pt o zona complet goala (hartie alba, perete), dand o eticheta
        # fals-pozitiva fara nicio proba independenta; (2) daca YOLO
        # CONFUNDA un semn de viteza real cu alta clasa (ii ghiceste gresit
        # PICTOGRAMA), calea rapida l-ar "fura" de la pipeline-ul robust
        # CNN+OCR care stie sa-l corecteze, blocand o eticheta gresita.
        # Verificarea forma+culoare respinge ambele cazuri: (1) nu gaseste
        # nicio forma/culoare valida -> refuza calea rapida; (2) gaseste o
        # forma/culoare care NU corespunde categoriei asteptate pt clasa
        # zisa de YOLO (ex. semnul e de fapt un cerc rosu, nu triunghiul pe
        # care-l astepta YOLO) -> refuza calea rapida, lasa CNN-ul sa decida.
        mapped_gtsrb = YOLO_CLASS_TO_GTSRB.get(yolo_cls_id)
        category_for_yolo_guess = (
            shape_color_to_category(shape_name, color_name)
            if shape_name is not None and color_name is not None else None
        )
        # Familia "rotund rosu" (cerc SAU octogon, nu putem distinge fiabil
        # intre ele -- vezi likely_classes_for_category) NU intra NICIODATA
        # pe calea rapida, chiar daca YOLO e sigur pe clasa lui: verificat
        # direct pe un cadru real din videoclipul de test, YOLO a confundat
        # un semn REAL de 130 km/h (cerc rosu) cu "stop", cu 52% incredere!
        # Toata familia asta (limite de viteza, interdictii, STOP) ramane pe
        # calea lenta/robusta (CNN + OCR), care e special construita si
        # testata sa distinga fin intre semne rotunde rosii -- exact ce
        # YOLO singur nu poate face de incredere.
        category_needs_slow_path = category_for_yolo_guess in ("circle_red", "octagon_red")
        yolo_guess_confirmed_by_shape_color = (
            category_for_yolo_guess is not None
            and not category_needs_slow_path
            and mapped_gtsrb in likely_classes_for_category(category_for_yolo_guess)
        )
        if (mapped_gtsrb is not None
                and yolo_cls_id not in SPEED_YOLO_CLASS_IDS
                and mapped_gtsrb not in SPEED_LIMIT_CLASSES
                and yolo_conf_score >= YOLO_CLASS_CONF_THRESHOLD
                and yolo_guess_confirmed_by_shape_color):
            distance_m = estimate_distance_m(depth_map, (x1, y1, w, h))
            detections.append({
                "bbox": (x1, y1, w, h),
                "class_id": mapped_gtsrb,
                "label_ro": label_for_class(mapped_gtsrb),
                "confidence": float(yolo_conf_score),
                "yolo_confidence": float(yolo_conf_score),
                "shape": shape_name,
                "color": color_name,
                "is_small": is_small,
                "confirm_frames_needed": CONFIRM_FRAMES_FAR if is_small else CONFIRM_FRAMES_NEAR,
                "distance_m": distance_m,
            })
            continue

        # --- Calea VECHE (neschimbata): limite de viteza + rezerva pt ------
        # clase YOLO nemapate/nesigure (CNN mic + shape/color + OCR) --------

        padding_ratio = BOX_PADDING_RATIO_SMALL if is_small else BOX_PADDING_RATIO

        pad = int(padding_ratio * max(w, h))
        px0, py0 = max(0, x1 - pad), max(0, y1 - pad)
        px1, py1 = min(w_img, x2 + pad), min(h_img, y2 + pad)

        roi = frame_bgr[py0:py1, px0:px1]
        if roi.size == 0:
            continue

        cls_id, conf = classifier.predict(roi)

        # shape_name/color_name au fost DEJA calculate mai sus (o singura
        # data, refolosite si de calea rapida) -- doar le aplicam ca
        # penalizare pe decizia CNN-ului, ca inainte.
        if use_shape_prior and box_big_enough:
            if shape_name is None or color_name is None:
                conf *= SHAPE_COLOR_UNKNOWN_PENALTY
            else:
                category = shape_color_to_category(shape_name, color_name)
                if category is None:
                    conf *= SHAPE_COLOR_NO_MATCH_PENALTY
                elif cls_id not in likely_classes_for_category(category):
                    conf *= SHAPE_COLOR_PENALTY

        if ENABLE_SPEED_OCR_OVERRIDE and (
            (shape_name == "circle" and color_name == "red") or cls_id in SPEED_LIMIT_CLASSES
        ):
            cls_id, conf, ocr_used = classify_speed_limit(
                roi, roi_strict, cls_id, conf, min_digit_conf=SPEED_OCR_MIN_DIGIT_CONF
            )
            if ocr_used:
                # citirea directa a cifrelor conteaza mai mult decat un
                # simplu vot CNN in istoricul tracker-ului (vezi
                # OCR_VOTE_BOOST mai sus) -- asa converge rapid la valoarea
                # corecta chiar daca au fost deja cateva cadre gresite de
                # la CNN inainte sa prindem un citit OCR clar
                conf = min(1.0, conf * OCR_VOTE_BOOST)

        if conf < cls_conf:
            continue

        # --- 4) distanta pana la semn (adancime Z), din harta ZED ----------
        distance_m = estimate_distance_m(depth_map, (x1, y1, w, h))

        detections.append({
            "bbox": (x1, y1, w, h),
            "class_id": cls_id,
            "label_ro": label_for_class(cls_id),
            "confidence": conf,
            "yolo_confidence": float(yolo_conf_score),
            "shape": shape_name,
            "color": color_name,
            "is_small": is_small,
            "confirm_frames_needed": CONFIRM_FRAMES_FAR if is_small else CONFIRM_FRAMES_NEAR,
            "distance_m": distance_m,
        })

    detections.sort(key=lambda d: -(d["bbox"][2] * d["bbox"][3]))
    for i, d in enumerate(detections):
        d["position"] = i
    return detections


def draw_hybrid_detections(frame_bgr, detections):
    """Deseneaza chenarele + eticheta STABILA (venita din tracker, nu bruta)."""
    out = frame_bgr.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        color = (0, 200, 0) if det.get("locked") else (0, 220, 255)

        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)

        lock_tag = " [LOCK]" if det.get("locked") else ""
        distance_m = det.get("distance_m")
        distance_tag = f" - {distance_m:.2f}m" if distance_m is not None else " - ?m"
        text = f"{det['label_ro']} ({det['confidence']*100:.0f}%){distance_tag}{lock_tag}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x, max(0, y - th - 10)), (x + tw + 6, y), color, -1)
        cv2.putText(out, text, (x + 3, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return out

# MAIN (camera) -- deschisa prin SDK-ul ZED, ca sa avem si harta de adancime

def open_zed_camera():
    """Deschide camera ZED 2 prin SDK-ul oficial (pyzed.sl), NU prin
    cv2.VideoCapture (acces UVC brut, cum era inainte) -- doar SDK-ul poate
    calcula harta de adancime (distanta pana la semn) din perechea stereo."""
    print("Pornim camera ZED 2 (prin SDK)...")
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = ZED_RESOLUTION
    init_params.camera_fps = FRAME_FPS
    init_params.depth_mode = ZED_DEPTH_MODE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = ZED_DEPTH_MIN_M
    init_params.depth_maximum_distance = ZED_DEPTH_MAX_M

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Eroare: nu pot deschide camera ZED 2 ({status}). "
              "Verifica daca e conectata pe USB si daca alt proces n-o mai foloseste.")
        sys.exit(1)

    cam_info = zed.get_camera_information()
    res = cam_info.camera_configuration.resolution
    print(f"Camera ZED 2 pornita -- rezolutie pe ochi: {res.width}x{res.height} "
          f"@ {cam_info.camera_configuration.fps:.0f} FPS, mod adancime: {ZED_DEPTH_MODE}.")
    return zed


def main():
    detector, classifier = load_models()
    zed = open_zed_camera()
    tracker = SignTracker()

    runtime_params = sl.RuntimeParameters()
    image_mat = sl.Mat()
    depth_mat = sl.Mat()

    prev_time = time.time()
    frame_idx = 0

    try:
        while True:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue

            # imaginea stanga, DEJA rectificata de SDK (spre deosebire de
            # cv2.VideoCapture pe indexul brut, care dadea perechea stereo
            # nerectificata, taiata manual la jumatate)
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            cadru_stang = cv2.cvtColor(image_mat.get_data(), cv2.COLOR_BGRA2BGR)

            # harta de adancime (metri), aliniata pixel-cu-pixel cu imaginea stanga
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth_map = depth_mat.get_data()

            detectii_brute = detect_and_classify_hybrid(
                cadru_stang, detector, classifier, frame_idx=frame_idx, depth_map=depth_map,
            )
            frame_idx += 1
            detectii_stabile = tracker.update(detectii_brute)
            cadru_final = draw_hybrid_detections(cadru_stang, detectii_stabile)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(cadru_final, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Sistem Hibrid - YOLO + MobileNetV2", cadru_final)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()