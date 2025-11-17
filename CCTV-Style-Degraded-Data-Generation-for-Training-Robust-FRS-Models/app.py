
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from Degradation.degradation import degrade_with_group
from Degradation.mtcnn_crop import detect_and_crop_face
from infer_frs import predict_image  # Import prediction function

# A dictionary to store custom text for each model
model_descriptions = {
    "HQ Model": {
        "training": "Trained on Only High Quality Images",
        "hq_accuracy": "Training Accuracy: 100%",
        "degraded_accuracy": "Testing Accuracy :99%"
    },
    "Degraded Dataset Model": {
        "training": "Trained on a mix of High Quality and Degraded Images",
        "hq_accuracy": " Training Accuracy: 89.59%",
        "degraded_accuracy": "Testing Accuracy: 70.27%"
    },
    "Slightly lesser Degradation Model": {
        "training": "Trained on a lightly degraded dataset",
        "hq_accuracy": "Training Accuracy: 97.54%",
        "degraded_accuracy": "Testing Accuracy: 83.9951%"
    },
    "Retrained Degraded Model": {
        "training": "Fine-tuned on a full spectrum degraded dataset",
        "hq_accuracy": "Training Accuracy: 99.94%",
        "degraded_accuracy": "Testing Accuracy: 87.37%"
    }
}

# ------------------ Streamlit App UI ------------------
st.set_page_config(page_title="Face Recognition System UI", layout="wide")
st.title("Image Degradation & Face Recognition")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("Upload & Settings")

    if st.button("Clear"):
        st.session_state.clear()
        st.rerun()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    degradation_group = st.selectbox(
        "Apply Degradation Group",
        ['Original (No degradation)',
         'Acquisition & Sensor Defects',
         'Transmission & Compression artifact',
         'Blurring Effects',
         'Noise Injection',
         'Acquisition & Sensor Defects + Transmission & Compression artifact',
         'Acquisition & Sensor Defects + Transmission & Compression artifact + Blurring Effects',
         'All'],
        index=0
    )

    st.markdown("---")
    st.subheader("Face Recognition")
    recognize_faces = st.button("Recognize Faces")

if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

if uploaded_file is not None:
    st.session_state["uploaded_file"] = uploaded_file

if st.session_state["uploaded_file"] is None:
    st.markdown("""
    <div style="text-align: center; padding: 40px; border-radius: 10px;">
        <h3>Welcome!</h3>
        <p>Upload an image and apply degradation groups for FRS testing.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    image = Image.open(st.session_state["uploaded_file"]).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    if degradation_group == 'Original (No degradation)':
        degraded_bgr = image_bgr.copy()
    else:
        degraded_bgr = degrade_with_group(image_bgr, degradation_group)

    original_with_bbox, cropped_degraded = detect_and_crop_face(image_bgr, degraded_bgr)

    degraded_rgb = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB)
    original_bbox_rgb = cv2.cvtColor(original_with_bbox, cv2.COLOR_BGR2RGB)
    degraded_image = Image.fromarray(degraded_rgb)
    original_bbox_image = Image.fromarray(original_bbox_rgb)

    resized_original = image.copy()
    resized_degraded = degraded_image.copy()

    cropped_pil = None
    if cropped_degraded is not None and cropped_degraded.size != 0:
        cropped_rgb = cv2.cvtColor(cropped_degraded, cv2.COLOR_BGR2RGB)
        cropped_pil = Image.fromarray(cropped_rgb)
    else:
        resized_cropped = None

    st.subheader("Image Comparisons")

    col1, col2 = st.columns(2)

    with col1:
        st.image(resized_original, caption="Original", use_container_width=200)

    with col2:
        st.image(resized_degraded, caption=f"Degraded: {degradation_group}", use_container_width=200)

    st.markdown("---")
    if recognize_faces and cropped_pil is not None:
        model_names = ["HQ Model", "Degraded Dataset Model", "Slightly lesser Degradation Model",
                       "Retrained Degraded Model"]

        for model_name in model_names:
            pred_crop, conf_crop = predict_image(cropped_pil, model_name)

            # --- Get the specific text for this model ---
            description = model_descriptions.get(model_name, {})

            # --- Use columns to display the image and prediction side-by-side ---
            col_img, col_pred = st.columns([1, 2])

            with col_img:
                st.subheader(model_name)
                st.image(cropped_pil, caption=f"{model_name} Input", width=170)
            with col_pred:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                # Custom color container (light blue example)
                custom_html = f"""
                <div style="
                    background-color: #66BB6A;  /* Change this color */
                    padding: 15px;
                    border-radius: 10px;
                    border: 1px solid #89b4f8;
                    color:black;
                ">
                    <strong>[{model_name}] Predicted from Cropped Face</strong>: {pred_crop} <br>
                    <strong>Confidence</strong>: {conf_crop:.2f}%
                </div>
                """
                st.markdown(custom_html, unsafe_allow_html=True)

                # Custom CSS for the description block
                st.markdown(
                    """
                    <style>
                    .description-box {
                        background-color: #FFF59D; /* Light blue background */
                        padding: 15px;
                        border-radius: 10px;
                        border: 1px solid #c3d0ff;
                        color:black;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Display block with color
                with st.container():
                    st.subheader("Model Description")
                    st.markdown(
                        f"""
                        <div class="description-box">
                            <b>{description.get('training', '')}</b><br>
                            <b>{description.get('hq_accuracy', '')}</b><br>
                            <b>{description.get('degraded_accuracy', '')}</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


    elif recognize_faces:
        st.warning("No cropped face available for recognition.")
