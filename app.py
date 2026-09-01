import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
from PIL import Image
import pytesseract

# tell pytesseract where Tesseract is installed (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- page setup ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection from Screenshots")
st.write("Upload a screenshot of a social media post or news headline. "
         "The app extracts the text using OCR and predicts whether it is Real or Fake.")

# --- load model once ---
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("./welfake_distilbert")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# --- prediction ---
def predict(text):
    inputs = tokenizer(text, truncation=True, padding=True,
                       max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred = probs.argmax().item()
    confidence = probs[pred].item()
    label = "REAL" if pred == 1 else "FAKE"
    return label, confidence

# --- upload + process ---
uploaded = st.file_uploader("Upload a screenshot", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded screenshot", use_container_width=True)

    extracted_text = pytesseract.image_to_string(image)
    st.subheader("Extracted text (OCR):")
    st.write(extracted_text if extracted_text.strip() else "_No text detected._")

    if extracted_text.strip():
        label, conf = predict(extracted_text)
        st.subheader("Prediction:")
        if label == "FAKE":
            st.error(f"🚫 {label}  (confidence: {conf:.1%})")
        else:
            st.success(f"✅ {label}  (confidence: {conf:.1%})")