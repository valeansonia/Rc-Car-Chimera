GTSRB_LABELS_RO = {
    0: "Limita 20 km/h", 1: "Limita 30 km/h", 2: "Limita 50 km/h", 3: "Limita 60 km/h",
    4: "Limita 70 km/h", 5: "Limita 80 km/h", 6: "Sfarsit limita 80 km/h", 7: "Limita 100 km/h",
    8: "Limita 120 km/h", 9: "Depasirea interzisa", 10: "Depasire interzisa (camioane)",
    # ATENTIE: clasa 11 in GTSRB standard e "Intersectie cu drum fara
    # prioritate" (triunghi de avertizare), NU "Trecere de pietoni" cum era
    # gresit inainte -- eticheta veche era pur si simplu incorecta si te
    # facea sa crezi ca modelul greseste, cand de fapt doar afisa numele
    # gresit pt o clasificare corecta.
    11: "Intersectie cu drum fara prioritate",
    12: "Drum cu prioritate",
    # Lipsea complet din dictionar, desi e deja folosita mai jos in
    # SHAPE_TO_LIKELY_CLASSES ca si cum ar exista -- fara ea, o clasificare
    # CORECTA a CNN-ului pt semnul de cedare trecere afisa doar "clasa 13"
    # (fallback-ul generic din label_for_class), nu un text real.
    13: "Cedeaza trecerea",
    14: "STOP", 15: "Acces interzis", 16: "Interzis autovehicule", 17: "Sens interzis",
    18: "Alte pericole (Atentie)", 19: "Curba stanga", 20: "Curba dreapta", 21: "Curba dubla",
    22: "Drum denivelat", 23: "Drum alunecos", 24: "Ingustare drum dreapta", 25: "Lucrari",
    26: "Semafoare", 27: "Pietoni", 28: "Copii", 29: "Biciclisti", 30: "Zapada/Gheata",
    31: "Animale salbatice", 32: "Sfarsit toate restrictiile", 33: "La dreapta",
    34: "La stanga", 35: "Inainte", 36: "Inainte sau la dreapta", 37: "Inainte sau la stanga",
    38: "Ocolire prin dreapta", 39: "Ocolire prin stanga", 40: "Sens giratoriu",
    41: "Sfarsit depasire interzisa", 42: "Sfarsit depasire interzisa (camioane)",
}

# Clase GTSRB care sunt limite de viteza -- le tratam separat in tracker
# pentru ca se confunda cel mai des intre ele (30/50/60/80/100/120), mai
# ales de la distanta, cand cifrele sunt mici si neclare in imagine.
#
# ATENTIE: GTSRB (setul pe care a fost antrenat clasificatorul) are DOAR
# aceste 8 valori de viteza. Limite folosite frecvent in Romania -- 90
# (extravilan), 110, 130 (autostrada) -- NU EXISTA ca si clase in model.
# CNN-ul e deci obligat sa "ghiceasca" cea mai apropiata clasa cunoscuta
# (90 -> 80, 130 -> 120). Vezi read_speed_limit_number() din
# sign_detector.py pentru corectia prin citire directa a cifrelor.
SPEED_LIMIT_CLASSES = {0, 1, 2, 3, 4, 5, 7, 8}

# Valori reale de viteza pe care le poate returna clasa GTSRB corespunzatoare
GTSRB_CLASS_TO_SPEED = {0: 20, 1: 30, 2: 50, 3: 60, 4: 70, 5: 80, 7: 100, 8: 120}

# Toate valorile de viteza plauzibile pe care le putem intalni pe drum la
# noi (folosit ca sa validam ce a "citit" OCR-ul de cifre, ca sa nu
# acceptam un numar aiurea daca segmentarea a mers prost)
PLAUSIBLE_SPEED_VALUES = {20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130}

# "Grupuri de confuzie" -- valorile de viteza pe care CNN-ul (si/sau
# vocabularul GTSRB, care nu are deloc 90/110/130) le confunda cel mai des
# INTRE ELE: 60/80/90 la viteze mici, 100/110/120/130 la viteze mari.
# In loc sa reantrenam un model separat pt fiecare grup (nu avem acces local
# la setul de date/Colab-ul unde a fost antrenat modelul original), tratam
# aceste grupuri ca pe niste "clasificatoare specializate" implementate prin
# citirea directa a cifrelor (OCR, vezi classify_speed_limit() din
# sign_detector.py): cand semnul detectat pare sa fie o limita de viteza din
# oricare din aceste grupuri, NU ne mai bazam pe eticheta bruta a CNN-ului
# (nesigur exact pe aceste perechi), ci pe numarul citit direct din imagine.
SPEED_CONFUSION_GROUP_LOW = {60, 80, 90}
SPEED_CONFUSION_GROUP_HIGH = {100, 110, 120, 130}
SPEED_CONFUSION_GROUPS = (SPEED_CONFUSION_GROUP_LOW, SPEED_CONFUSION_GROUP_HIGH)


def speed_confusion_group(speed):
    """Returneaza grupul de confuzie din care face parte `speed`, sau None
    daca valoarea nu e intr-un grup cunoscut ca problematic (ex. 20/30/50)."""
    for group in SPEED_CONFUSION_GROUPS:
        if speed in group:
            return group
    return None

SHAPE_TO_LIKELY_CLASSES = {
    "octagon_red": {14}, # STOP
    "triangle_red": {11, 13, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}, # Avertizare / Cedeaza
    "circle_red": {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 17}, # Interzicere
    "circle_blue": {33, 34, 35, 36, 37, 38, 39, 40}, # Obligare
    "diamond_yellow": {12} # Drum cu prioritate
}

# "octagon_red" si "circle_red" sunt tratate ca interschimbabile la
# verificarea de potrivire forma+culoare. Motiv (verificat direct, nu
# teoretic): distinctia geometrica cerc-vs-octogon prin approxPolyDP nu e
# fiabila la rezolutiile tipice ale unui decupaj de semn -- un octogon
# REGULAT (STOP) are circularitate ~0.948, iar un cerc real, pixelat, la
# rezolutie mica poate iesi cu circularitate ~0.89 SI acelasi numar de
# "colturi" aproximate (~8) ca octogonul -- valorile se suprapun, deci
# niciun prag fix nu le separa curat. Mai bine acceptam ambiguitatea decat
# sa respingem un STOP real doar pt ca euristica l-a numit "circle" (sau
# invers, un cerc real numit "octagon").
_RED_ROUND_CATEGORIES = ("circle_red", "octagon_red")


def likely_classes_for_category(category):
    """La fel ca SHAPE_TO_LIKELY_CLASSES.get(category, set()), dar pt
    'circle_red'/'octagon_red' returneaza reuniunea ambelor categorii (vezi
    explicatia de mai sus)."""
    if category in _RED_ROUND_CATEGORIES:
        classes = set()
        for cat in _RED_ROUND_CATEGORIES:
            classes |= SHAPE_TO_LIKELY_CLASSES.get(cat, set())
        return classes
    return SHAPE_TO_LIKELY_CLASSES.get(category, set())


def label_for_class(class_id):
    """Traduce un class_id in eticheta afisata pe ecran.

    class_id >= 0  -> clasa GTSRB normala (cautata in GTSRB_LABELS_RO)
    class_id <  0  -> "id virtual" folosit de citirea OCR a cifrelor pt.
                       o limita de viteza care NU exista in GTSRB (ex.
                       90/110/130 km/h). Conventia: class_id = -viteza,
                       deci -90 inseamna "Limita 90 km/h".
    """
    if class_id < 0:
        return f"Limita {-class_id} km/h"
    return GTSRB_LABELS_RO.get(class_id, f"clasa {class_id}")