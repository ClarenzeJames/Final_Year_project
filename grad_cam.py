import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

import matplotlib.pyplot as plt
import matplotlib as mpl

import whole_model
WholeModel = whole_model.Whole_model()

whole_model_save_file = "./best_hybrid_whole_model.weights.h5"
image_path = "../../archive/full mammogram images/test/benign/P_00007_calcification_2_test.jpg"
last_conv_layer_name = "top_conv"

def load_rgb_for_model(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(img, (384,384), interpolation=cv2.INTER_AREA)
    x = preprocess_input(rgb.astype(np.float32))
    x = np.expand_dims(x, axis=0)

    return rgb, x

def overlay_heatmap(rgb, heatmap, alpha=0.35, cmap=cv2.COLORMAP_JET):
    hm = np.uint8(255 * np.clip(heatmap, 0, 1))
    hm = cv2.applyColorMap(hm, cmap)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(rgb, 1.0, hm, alpha, 0.0)
    

def make_grad_model(model, last_conv_layer_name):
    last_conv = model.get_layer(last_conv_layer_name).output
    preds = model.output
    return tf.keras.Model(model.inputs, [last_conv, preds])

@tf.function
def gradcam_score_and_map(grad_model, x):
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        score = preds[:, 0]

    grads = tape.gradient(score, conv_out)

    weights = tf.reduce_mean(grads, axis=(1,2), keepdims=True)
    cam = tf.reduce_sum(weights * conv_out, axis=-1)[0]
    cam = tf.nn.relu(cam)

    cam = (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) + 1e-8)

    return preds[0,0], cam

two_view = WholeModel.build_whole_model()
two_view.load_weights(whole_model_save_file)
two_view.summary()

rgb, x = load_rgb_for_model(image_path)

grad_model = make_grad_model(two_view, last_conv_layer_name)
prob, cam = gradcam_score_and_map(grad_model, x)

cam_up = cv2.resize(cam.numpy(), (384, 384), interpolation=cv2.INTER_CUBIC)
overlay = overlay_heatmap(rgb, cam_up, alpha=0.35)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(rgb)
plt.title("Input")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cam_up, cmap="jet")
plt.title("Grad-CAM")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(overlay)
plt.title(f"Overlay (p={float(prob):.3f})")
plt.axis("off")

plt.tight_layout()
plt.show()



