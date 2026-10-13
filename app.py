import streamlit as st
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from model import AgeCNN

# OpenCV load wenne nathnam app eka crash wenna denne na
try:
    import cv2
    HAS_CV2 = hasattr(cv2, "CascadeClassifier")
except Exception:
    HAS_CV2 = False

AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39",
              "40-49", "50-59", "60-69", "70+"]

CONFIDENCE_THRESHOLD = 0.40

st.set_page_config(page_title="FairVision", page_icon="🧑", layout="wide")

# ---------------------- Styling ----------------------
CSS = """
<style>
.hero {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    padding: 30px 20px;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 22px;
}
.hero h1 { margin: 0; font-size: 2.4rem; color: white; }
.hero p { margin: 8px 0 0 0; opacity: 0.93; font-size: 1.05rem; }

.card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 14px;
}
.welcome { text-align: center; padding: 34px 20px; }
.welcome h3 { margin-bottom: 8px; }
.steps { display: flex; gap: 14px; justify-content: center; flex-wrap: wrap; margin-top: 18px; }
.step {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-radius: 14px;
    padding: 14px 18px;
    min-width: 170px;
}

.top-pred { text-align: center; }
.top-caption { opacity: 0.7; font-size: 0.9rem; margin-bottom: 0; }
.top-label {
    font-size: 3.4rem;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(90deg, #818cf8, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.top-conf { font-size: 1.15rem; margin-top: 4px; }

.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 8px;
}
.badge-high { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }
.badge-mid  { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
.badge-low  { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }

.row-head { display: flex; justify-content: space-between; margin-bottom: 4px; font-weight: 600; }
.bar-wrap { background: rgba(255, 255, 255, 0.1); border-radius: 999px; height: 12px; overflow: hidden; margin-bottom: 14px; }
.bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #6366f1, #ec4899); }

.stat {
    text-align: center;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 14px;
    padding: 14px 8px;
}
.stat b { font-size: 1.5rem; display: block; }
.stat span { opacity: 0.7; font-size: 0.85rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------- Model + face detector ----------------------
@st.cache_resource
def load_model():
    model = AgeCNN()
    model.load_state_dict(torch.load("mit_oversample_best.pt", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_face_detector():
    if not HAS_CV2:
        return None
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        det = cv2.CascadeClassifier(path)
        return None if det.empty() else det
    except Exception:
        return None

model = load_model()
detector = load_face_detector()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def crop_face(pil_image, margin=0.25):
    """Find the largest face and crop it with a margin. Returns None if not possible."""
    if detector is None:
        return None
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_w, pad_h = int(w * margin), int(h * margin)
    x1, y1 = max(x - pad_w, 0), max(y - pad_h, 0)
    x2, y2 = min(x + w + pad_w, img.shape[1]), min(y + h + pad_h, img.shape[0])
    return Image.fromarray(img[y1:y2, x1:x2])

def predict_with_flip(face):
    """Average the predictions of the original and the mirrored image."""
    x = transform(face).unsqueeze(0)
    x_flip = torch.flip(x, dims=[3])
    with torch.no_grad():
        p1 = torch.softmax(model(x), dim=1)[0]
        p2 = torch.softmax(model(x_flip), dim=1)[0]
    return (p1 + p2) / 2

def confidence_badge(conf):
    if conf >= 0.60:
        return '<span class="badge badge-high">High confidence</span>'
    if conf >= CONFIDENCE_THRESHOLD:
        return '<span class="badge badge-mid">Medium confidence</span>'
    return '<span class="badge badge-low">Low confidence</span>'

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.markdown("## 🧑 FairVision")
    st.write("A CNN trained from scratch in PyTorch to predict the age group of a face, "
             "with a focus on fairness across race and gender.")
    st.markdown("---")
    st.markdown("**Model details**")
    st.write("- Dataset: FairFace (0.25 crop)")
    st.write("- Training subset: 20,000 images")
    st.write("- Input size: 128 x 128")
    st.write("- Mitigation: race x age balanced oversampling")
    st.markdown("---")
    st.caption("Educational prototype. Not for real-world decisions.")

# ---------------------- Header ----------------------
st.markdown(
    """
    <div class="hero">
        <h1>FairVision</h1>
        <p>Age group prediction with a bias-aware CNN. Upload a face photo and see the top 3 predictions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_how, tab_resp = st.tabs(["🔍 Predict", "⚙️ How it works", "⚠️ Responsible use"])

# ---------------------- Tab 1: Predict ----------------------
with tab_predict:
    uploaded = st.file_uploader(
        "Upload a face image (JPG or PNG)", type=["jpg", "jpeg", "png"]
    )
    st.caption("Tip: use a clear, front-facing photo of one person, with the face filling most of the image.")

    if uploaded is None:
        st.markdown(
            """
            <div class="card welcome">
                <h3>👋 Welcome!</h3>
                <p>Upload a photo to see which age group the model predicts.</p>
                <div class="steps">
                    <div class="step"><b>1. Upload</b><br>Choose a face photo</div>
                    <div class="step"><b>2. Detect</b><br>The face is cropped automatically</div>
                    <div class="step"><b>3. Predict</b><br>See the top 3 age groups</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        image = Image.open(uploaded).convert("RGB")
        face = crop_face(image)
        face_found = face is not None
        if not face_found:
            face = image

        probs = predict_with_flip(face)
        top_probs, top_idx = torch.topk(probs, 3)
        top_conf = top_probs[0].item()

        col_img, col_res = st.columns([1, 1.2], gap="large")

        with col_img:
            st.markdown("##### Your photo")
            st.image(image, use_container_width=True)
            if face_found:
                st.markdown("##### Detected face (used for prediction)")
                st.image(face, width=200)
            else:
                st.warning(
                    "Automatic face cropping was not possible, so the whole image is used. "
                    "Results may be unreliable. Try a closely cropped, front-facing photo."
                )

        with col_res:
            st.markdown(
                f"""
                <div class="card top-pred">
                    <p class="top-caption">Predicted age group</p>
                    <div class="top-label">{AGE_LABELS[top_idx[0].item()]}</div>
                    <div class="top-conf">{top_conf * 100:.1f}% confidence</div>
                    {confidence_badge(top_conf)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("##### Top 3 predictions")
            bars_html = '<div class="card">'
            for rank, (p, i) in enumerate(zip(top_probs, top_idx), start=1):
                pct = p.item() * 100
                bars_html += (
                    f'<div class="row-head"><span>{rank}. {AGE_LABELS[i.item()]}</span>'
                    f'<span>{pct:.1f}%</span></div>'
                    f'<div class="bar-wrap"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>'
                )
            bars_html += "</div>"
            st.markdown(bars_html, unsafe_allow_html=True)
            st.caption("Prediction is averaged over the original and the mirrored image.")

            if top_conf < CONFIDENCE_THRESHOLD:
                st.warning(
                    f"The model is not confident about this photo (top confidence is below "
                    f"{int(CONFIDENCE_THRESHOLD * 100)}%). Please treat this result as a rough guess only."
                )

# ---------------------- Tab 2: How it works ----------------------
with tab_how:
    st.markdown("### How it works")
    st.write(
        "This demo uses a CNN trained from scratch in PyTorch on the FairFace dataset "
        "to predict the age group of a face image. The model was trained with "
        "race-by-age balanced oversampling to reduce demographic bias."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="stat"><b>9</b><span>age groups</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="stat"><b>~39%</b><span>test accuracy</span></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="stat"><b>4</b><span>conv blocks</span></div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown(
        "**Pipeline:** upload photo → detect and crop the face → resize to 128x128 → "
        "CNN predicts probabilities for 9 age groups → show the top 3."
    )

# ---------------------- Tab 3: Responsible use ----------------------
with tab_resp:
    st.markdown("### Limitations and responsible use")
    st.write(
        "- Overall test accuracy is about 39%, so predictions can be wrong, "
        "especially between neighbouring age groups.\n"
        "- Performance still differs across race and gender groups.\n"
        "- The model was trained on tightly cropped face images, so photos with "
        "different framing, lighting, or angles may give unreliable results.\n"
        "- Trained on a 20,000-image subset of FairFace with limited training.\n"
        "- This is an educational prototype. Do not use it for identity, security, "
        "hiring, or any decision that affects people."
    )
