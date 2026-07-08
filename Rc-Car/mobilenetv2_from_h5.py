"""
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
    return model