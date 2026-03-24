import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Lung Disease Detection",
    page_icon="🫁",
    layout="wide"
)

# =========================================================
# CSS (COMPACT & CLEAN)
# =========================================================
st.markdown("""
<style>
.block-container{
    max-width:1100px;
    padding-top:1.2rem;
}
img{
    border-radius:12px;
    max-height:260px;
    object-fit:contain;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_ai():
    model = load_model("models/resnet50_best_model.h5")
    classes = ['COVID', 'Normal', 'Viral_Pneumonia']
    return model, classes

MODEL, CLASS_NAMES = load_ai()

# =========================================================
# PREPROCESS (SESUIA TRAINING)
# =========================================================
def preprocess(img):
    img = img.convert("RGB")
    img.thumbnail((512,512))
    img = img.resize((224,224))
    img = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# =========================================================
# PREDICT
# =========================================================
def predict(img):
    batch = preprocess(img.copy())
    preds = MODEL.predict(batch, verbose=0)[0]
    idx = np.argmax(preds)
    conf = float(preds[idx] * 100)
    return idx, conf, preds, batch

# =========================================================
# GRAD-CAM
# =========================================================
def get_last_conv(model):
    for layer in reversed(model.layers):
        if "conv" in layer.name:
            return layer.name
    return None

LAST_CONV = get_last_conv(MODEL)

def gradcam(img_array, model, idx):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(LAST_CONV).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv, preds = grad_model(img_array)
        loss = preds[:, idx]

    grads = tape.gradient(loss, conv)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    conv = conv[0]

    heatmap = conv @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

def overlay(original, heatmap):
    original = np.array(original).astype(np.uint8)

    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

# =========================================================
# HEADER
# =========================================================
st.title("🫁 AI Lung Disease Detection")
st.caption("Chest X-Ray Classification using ResNet-50")

# =========================================================
# UPLOAD
# =========================================================
files = st.file_uploader(
    "Upload Chest X-Ray (multi file)",
    type=["png","jpg","jpeg"],
    accept_multiple_files=True
)

st.divider()

# =========================================================
# MAIN
# =========================================================
if files:
    progress = st.progress(0)

    for i, file in enumerate(files):
        progress.progress((i+1)/len(files))

        try:
            img = Image.open(file).convert("RGB")

            # SESSION STATE
            if f"res_{i}" not in st.session_state:
                st.session_state[f"res_{i}"] = None

            with st.expander(f"📁 {file.name}", expanded=False):

                # ===== TOP ROW =====
                col1, col2 = st.columns([1,1])

                # IMAGE (SMALL)
                with col1:
                    c_left, c_mid, c_right = st.columns([1,2,1])
                    with c_mid:
                        st.image(img, width=260)

                # RESULT PANEL
                with col2:
                    if st.button("🔍 Analyze", key=f"btn_{i}"):
                        with st.spinner("Analyzing..."):
                            st.session_state[f"res_{i}"] = predict(img)

                    if st.session_state[f"res_{i}"]:
                        idx, conf, preds, batch = st.session_state[f"res_{i}"]
                        label = CLASS_NAMES[idx]

                        if label == "Normal":
                            st.success("Normal")
                        elif label == "COVID":
                            st.error("COVID-19")
                        else:
                            st.warning("Viral Pneumonia")

                        st.markdown(f"### Confidence: {conf:.2f}%")
                        st.progress(conf/100)

                # ===== BOTTOM ROW =====
                if st.session_state[f"res_{i}"]:
                    idx, conf, preds, batch = st.session_state[f"res_{i}"]

                    st.divider()
                    colA, colB = st.columns([1,1])

                    # GRADCAM
                    with colA:
                        heatmap = gradcam(batch, MODEL, idx)
                        base = np.array(img.resize((224,224)))
                        result = overlay(base, heatmap)

                        st.image(result, caption="Grad-CAM (AI Focus Area)", use_container_width=True)

                    # PROBABILITIES
                    with colB:
                        st.markdown("**Class Probabilities**")
                        for j, p in enumerate(preds):
                            st.progress(float(p))
                            st.caption(f"{CLASS_NAMES[j]}: {p*100:.2f}%")

        except UnidentifiedImageError:
            st.error(f"{file.name} tidak valid")

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption("For research only • Not a medical diagnosis")