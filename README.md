# FairVision: Detecting and Mitigating Bias in a CNN-Based Age Group Classifier

**Live demo:** https://fairvision-demo-equyuahjjcnmz8ov6duzfm.streamlit.app/

FairVision is a CNN trained from scratch in PyTorch that predicts the **age group** of a face image using the **FairFace** dataset. The project audits the model for bias across **race** and **gender**, applies bias-mitigation strategies, and compares the results.

This is an individual assignment for the Certified AI & ML Engineer programme (IJSE).

- **Student:** [Malsha Nethmini]
- **Programme:** Certified AI & ML Engineer, Institute of Software Engineering (IJSE)

---

## What the demo does

Upload a face photo and the app will:

1. Detect and crop the face automatically (if OpenCV is available).
2. Predict the age group using the trained CNN.
3. Show the **top 3 predicted age groups** with their confidence.
4. Show a confidence badge and a warning when the model is not confident.

The prediction is averaged over the original image and its mirrored version to make it more stable.

Age groups: `0-2`, `3-9`, `10-19`, `20-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70+`

---

## Dataset

- **FairFace** (Hugging Face: `HuggingFaceM4/FairFace`, configuration `0.25`)
- Training subset: 20,000 images, stratified by age, split into 17,000 train and 3,000 validation (seed 42)
- Final test set: the official Hugging Face validation split (10,954 images)
- Target: age group (9 classes). Race and gender were used only for the fairness audit.

---

## Model

A custom CNN built from scratch (no pretrained models or transfer learning):

- 4 convolution blocks (32, 64, 128, 256 filters), each with Conv 3x3, BatchNorm, ReLU and MaxPool
- Global average pooling
- Fully connected classifier with Dropout (0.5 and 0.3)
- Input size: 128 x 128, about 423,000 parameters
- Optimizer: Adam (lr 0.001), loss: CrossEntropy, 25 epochs, batch size 64

---

## Results (test set, 10,954 images)

| Model | Overall accuracy | Macro F1 | Race gap | Worst race group acc | Gender gap |
|---|---|---|---|---|---|
| Baseline | 0.432 | 0.293 | 0.143 | 0.390 | 0.043 |
| Class-weighted loss | 0.335 | 0.295 | 0.079 | 0.296 | 0.028 |
| Race x age oversampling | 0.392 | 0.362 | 0.063 | 0.364 | 0.003 |

Gap = best group accuracy minus worst group accuracy.

**Bias mitigation strategies used:**
1. Class-weighted loss
2. Oversampling with a balanced sampler (race x age)

The **oversampling model** is the one used in the demo. It reduced the race and gender gaps and improved macro F1, at the cost of about 4 points of overall accuracy.

---

## Limitations and responsible use

- Overall accuracy is about 39%, so predictions can be wrong, especially between neighbouring age groups.
- Performance still differs across race and gender groups.
- The model was trained on tightly cropped face images, so other framing, lighting or angles may give unreliable results.
- Trained on a 20,000-image subset with limited training time.
- This is an educational prototype. **Do not use it** for identity, security, hiring, or any decision that affects people.

---

## Repository files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web application |
| `model.py` | CNN architecture (`AgeCNN`) |
| `mit_oversample_best.pt` | Trained model weights (oversampling model) |
| `requirements.txt` | Python dependencies |

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## References

- Karkkainen, K. and Joo, J. (2021). *FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation.* WACV 2021.
- FairFace on Hugging Face: https://huggingface.co/datasets/HuggingFaceM4/FairFace
- PyTorch: https://pytorch.org/
- Streamlit: https://streamlit.io/
