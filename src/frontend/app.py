import sys
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import time
import os
import face_recognition
import numpy as np

# Add root directory to sys.path so "src" becomes importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.backend.model.model import PlayerClassifier 

MODEL_PATH = "trained-models/resnet18-players.pth"
CLASS_IMAGES_DIR = "data/players-portraits"  

@st.cache_resource
def load_model():
    model = PlayerClassifier(data_dir="data/cropped-images")
    model.model.load_state_dict(torch.load(MODEL_PATH, map_location=model.device))
    model.model.eval()
    return model

# prediction logic
def crop_face_from_image(pil_image):
    image_np = np.array(pil_image)
    face_locations = face_recognition.face_locations(image_np)
    if len(face_locations) != 1:
        return None
    top, right, bottom, left = face_locations[0]
    face_image = image_np[top:bottom, left:right]
    return Image.fromarray(face_image)

def predict_image(image, model, class_names):
    cropped = crop_face_from_image(image)
    if cropped is None:
        return "No face or multiple faces detected", 0.0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(cropped).unsqueeze(0).to(model.device)

    with torch.no_grad():
        outputs = model.model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        print(probs)
    
    return class_names[pred.item()], conf.item() * 100

# streamlit frontend app
st.title("Bayern München Player Recognition")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("data/players-portraits/bayern-muenchen-logo.png", use_container_width=True)

st.write("This is a simple user interface to test a ML image classification model. So far, the model was trained " \
"on five different FC Bayern München players. For any image uploaded, it will predict what player the image most likely depicts." \
" The players, the model knows so far, are the following:") 

st.write("- Alphonso Davies") 
st.write("- Joshua Kimmich") 
st.write("- Leon Goretzka") 
st.write("- Michael Olise") 
st.write("- Thomas Müller") 

st.write("For best results, upload a high quality image that has only one player in it.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model = load_model()
    class_name, confidence = predict_image(image, model, model.class_names)

    st.write("🔍 Matching...")

    # animate slider
    progress = st.empty()
    for i in range(0, int(confidence) + 1):
        progress.progress(i / 100.0)
        time.sleep(0.02)
    progress.progress(confidence / 100.0)

    st.success(f"✅ Match: **{class_name}** ({confidence:.1f}%)")

    # show predicted player image
    player_img_path = os.path.join(CLASS_IMAGES_DIR, f"{class_name}.jpg")
    if os.path.exists(player_img_path):
        st.image(player_img_path, caption=class_name, width=300)
