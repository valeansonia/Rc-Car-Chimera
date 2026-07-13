"""
mobilenetv2_from_h5_feature_vector.py
--------------------------------------
Versiune ACTUALIZATA a scriptului tau original, adaptata pentru noul model
antrenat cu MobileNetV2 varianta "feature-vector" (in loc de "classification").

De ce a trebuit schimbat:
Varianta "classification" (folosita in modelul vechi) are un strat suplimentar
la final -- "MobilenetV2/Logits/Conv2d_1c_1x1" -- care proiecteaza cele 1280
caracteristici in 1001 clase ImageNet, urmat de un GlobalAveragePooling.
Varianta "feature-vector" (cea corecta pentru transfer learning) SE OPRESTE
inainte de acel strat: output-ul e direct vectorul de 1280 caracteristici
dupa GlobalAveragePooling pe "Conv_1". Deci acest script NU mai citeste/
reconstruieste stratul Logits -- daca modelul tau nou e antrenat cu
feature-vector si incerci sa incarci cu scriptul VECHI, va da eroare
(cheia "MobilenetV2/Logits/Conv2d_1c_1x1/weights:0" nu va exista in .h5).

Restul logicii (citire greutati direct din grupurile HDF5, fara acces la
internet / kaggle.com / storage.googleapis.com) ramane identica.
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


def _block(x, h5file, block_name, stride, block_idx):
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


_STRIDES = [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]


def build_mobilenetv2_feature_backbone(h5file, input_tensor):
    """Reconstruieste backbone-ul MobileNetV2 pana la vectorul de
    caracteristici (1280-d), FARA stratul final de clasificare ImageNet."""
    w0 = _g(h5file, "MobilenetV2/Conv/weights:0")
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name="stem_pad")(input_tensor)
    conv0 = layers.Conv2D(w0.shape[-1], 3, strides=2, padding="valid", use_bias=False, name="stem_conv")
    x = conv0(x)
    conv0.set_weights([w0])
    x = _bn(x, h5file, "MobilenetV2/Conv", "stem_bn")
    x = _relu6(x)

    for i, stride in enumerate(_STRIDES):
        block_name = "expanded_conv" if i == 0 else f"expanded_conv_{i}"
        x = _block(x, h5file, block_name, stride, i)

    w1 = _g(h5file, "MobilenetV2/Conv_1/weights:0")
    conv1 = layers.Conv2D(w1.shape[-1], 1, padding="same", use_bias=False, name="head_conv")
    x = conv1(x)
    conv1.set_weights([w1])
    x = _bn(x, h5file, "MobilenetV2/Conv_1", "head_bn")
    x = _relu6(x)

    # *** AICI SE OPRESTE varianta feature-vector ***
    # (nu mai exista Logits/Conv2d_1c_1x1 de reconstruit)
    x = layers.GlobalAveragePooling2D(name="head_gap")(x)
    return x  # (batch, 1280) -- vector de caracteristici, fara softmax


def _find_dense_layer_keys(h5file):
    """Gaseste automat numele stratului Dense final salvat de Keras,
    indiferent daca a fost numit dense, dense_1, final_dense etc."""
    for key in h5file.keys():
        if "dense" in key.lower():
            for sub in h5file[key].keys():
                kernel_path = f"{key}/{sub}/kernel:0"
                bias_path = f"{key}/{sub}/bias:0"
                if kernel_path in h5file:
                    return kernel_path, bias_path
    raise KeyError("Nu am gasit niciun strat Dense in fisierul .h5 "
                    "(cauta manual numele corect cu h5py.File(path,'r').keys())")


def build_full_model(h5_path, input_size=32):
    """Reconstruieste Resizing(224) -> Rescaling(1/255) -> MobileNetV2
    feature-vector (alpha=0.5) -> Dense(43, softmax), incarcand toate
    greutatile din fisierul .h5 original (fara acces la internet)."""
    with h5py.File(h5_path, "r") as h5file:
        inp = layers.Input(shape=(input_size, input_size, 3), name="input")
        x = layers.Resizing(224, 224, name="resizing")(inp)
        x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)
        feats = build_mobilenetv2_feature_backbone(h5file, x)

        dk_path, db_path = _find_dense_layer_keys(h5file)
        dk = _g(h5file, dk_path)
        db = _g(h5file, db_path)

        dense = layers.Dense(dk.shape[-1], activation="softmax", name="final_dense")
        out = dense(feats)
        dense.set_weights([dk, db])

    model = tf.keras.Model(inp, out)
    return model


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "model_semne_100_epoci.h5"
    model = build_full_model(model_path)
    model.summary()
    print("Model reconstruit cu succes din greutati, fara acces la internet.")