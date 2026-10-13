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

st.set_page_config(page_title="FairVision", page_icon="🧑")

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

st.title("FairVision: Age Group Classifier")
st.write(
    "This demo uses a CNN trained from scratch in PyTorch on the FairFace dataset "
    "to predict the age group of a face image. The model was trained with "
    "race-by-age balanced oversampling to reduce demographic bias."
)
st.info("Tip: use a clear, front-facing photo of one person, with the face filling most of the image.")

uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width=250)

    face = crop_face(image)
    if face is None:
        st.warning("Automatic face cropping was not possible, so the whole image is used. "
                   "Results may be unreliable. Try a closely cropped, front-facing photo.")
        face = image
    else:
        st.image(face, caption="Detected face (used for prediction)", width=200)

    x = transform(face).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    top_probs, top_idx = torch.topk(probs, 3)

    st.subheader("Top 3 predictions")
    for p, i in zip(top_probs, top_idx):
        st.write(f"**{AGE_LABELS[i]}**: {p.item() * 100:.1f}%")
        st.progress(float(p))

st.markdown("---")
st.subheader("Limitations and responsible use")
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
