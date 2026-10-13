import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import AgeCNN

AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39",
              "40-49", "50-59", "60-69", "70+"]

st.set_page_config(page_title="FairVision", page_icon="🧑")

@st.cache_resource
def load_model():
    model = AgeCNN()
    model.load_state_dict(torch.load("mit_oversample_best.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

st.title("FairVision: Age Group Classifier")
st.write(
    "This demo uses a CNN trained from scratch in PyTorch on the FairFace dataset "
    "to predict the age group of a face image. The model was trained with "
    "race-by-age balanced oversampling to reduce demographic bias."
)

uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width=250)

    x = transform(image).unsqueeze(0)
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
    "- Trained on a 20,000-image subset of FairFace with limited training.\n"
    "- This is an educational prototype. Do not use it for identity, security, "
    "hiring, or any decision that affects people."
)
