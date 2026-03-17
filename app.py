import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import cv2

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Lung Disease Detection",
    page_icon="🫁",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>

.block-container{
    max-width:1100px;
    padding-top:2rem;
}

.title{
    font-size:30px;
    font-weight:700;
}

.subtitle{
    color:#9ca3af;
    font-size:14px;
}

img{
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def init_server():
    model = load_model("models/resnet50_best_model.h5")
    classes = ['COVID', 'Normal', 'Viral_Pneumonia']
    return model, classes

MODEL_GLOBAL, CLASS_NAMES = init_server()

# =========================================================
# SESSION STATE
# =========================================================
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# =========================================================
# PREPROCESS
# =========================================================
def preprocess_image(img):

    img = img.convert("RGB")
    img = img.resize((224,224))

    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

# =========================================================
# PREDICTION
# =========================================================
def web_predict(img):
    img_batch = preprocess_image(img)
    preds = MODEL_GLOBAL.predict(img_batch, verbose=0)[0]
    idx = np.argmax(preds)
    confidence = float(preds[idx] * 100)
    return idx, confidence, img_batch

# =========================================================
# GRAD-CAM
# =========================================================
LAST_CONV_LAYER = "conv5_block3_out"

def make_gradcam_heatmap(img_array, model, pred_index=None):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap,0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap):

    heatmap = cv2.resize(heatmap,(original_img.shape[1],original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap * 0.4 + original_img

    return superimposed.astype(np.uint8)

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="title">🫁 AI Lung Disease Detection</div>', unsafe_allow_html=True)

st.markdown(
'<div class="subtitle">Chest X-Ray Classification using Transfer Learning ResNet-50</div>',
unsafe_allow_html=True
)

st.divider()

# =========================================================
# UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
"Upload Chest X-Ray Image",
type=["png","jpg","jpeg"]
)

# =========================================================
# MAIN UI
# =========================================================
if uploaded_file is not None:

    try:
        img = Image.open(uploaded_file)

        col1, col2 = st.columns([1.3,1])

        # =========================================
        # LEFT : IMAGE
        # =========================================
        with col1:

            st.markdown("### Chest X-Ray Image")
            st.image(img, use_container_width=True)

        # =========================================
        # RIGHT : RESULT
        # =========================================
        with col2:

            st.markdown("### AI Diagnosis")

            if st.button("🔍 Run AI Diagnosis", use_container_width=True):

                with st.spinner("AI analyzing lung patterns..."):

                    idx, confidence, img_batch = web_predict(img)

                    st.session_state.prediction = {
                        "idx": idx,
                        "confidence": confidence,
                        "img_batch": img_batch
                    }

            result = st.session_state.prediction

            if result is not None:

                diagnosis = CLASS_NAMES[result["idx"]]
                confidence = result["confidence"]

                if diagnosis == "Normal":
                    st.success("Paru-Paru Normal")
                else:
                    st.error("Terdeteksi Penyakit Paru")

                st.metric("Model Confidence", f"{confidence:.2f}%")

                # ==================================================
                # GAUGE
                # ==================================================
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    number={'suffix': "%"},
                    title={'text': "Model Confidence"},
                    gauge={
                        'axis': {'range': [0,100]},
                        'bar': {'color': "#3b82f6"}
                    }
                ))

                gauge.update_layout(
                    height=350,
                    margin=dict(l=20,r=20,t=40,b=10)
                )

                st.plotly_chart(gauge, use_container_width=True)

        # ==================================================
        # GRAD-CAM
        # ==================================================
        if st.session_state.prediction is not None:

            st.divider()

            st.markdown(
                "<h3 style='text-align:center;'>Grad-CAM Explainability</h3>",
                unsafe_allow_html=True
            )

            img_batch = st.session_state.prediction["img_batch"]

            heatmap = make_gradcam_heatmap(img_batch, MODEL_GLOBAL)

            original = np.array(img.resize((224,224)))

            superimposed = overlay_heatmap(original, heatmap)

            center1, col3, col4, center2 = st.columns([1,2,2,1])

            with col3:

                st.markdown(
                    "<p style='text-align:center;font-weight:600;'>Original X-Ray</p>",
                    unsafe_allow_html=True
                )

                st.image(original, width=350)

            with col4:

                st.markdown(
                    "<p style='text-align:center;font-weight:600;'>AI Attention (Grad-CAM)</p>",
                    unsafe_allow_html=True
                )

                st.image(superimposed, width=350)

    except UnidentifiedImageError:

        st.error("File bukan gambar valid")

# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption("AI system for research purposes only • Not medical diagnosis")