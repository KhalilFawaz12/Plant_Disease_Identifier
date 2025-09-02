import os
import sys
import streamlit as st
from PIL import Image
import numpy as np
import torch


# Add repo root to PATH so `import src...` works on hosted environments
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



from src.model import create_efficientnetb1_model
from src.predict import preprocess_image, predict_image


DEVICE ="cpu"
@st.cache_resource



def class_to_phrase(class_name):

    if class_name=="Pepper__bell___Bacterial_spot":
        return "Pepper bacterial spot"

    elif class_name=="Pepper__bell___healthy":
        return "Healthy bell pepper plant"

    elif class_name=="Potato___Early_blight":
        return "Potato early blight"

    elif class_name=="Potato___Late_blight":
        return "Potato late blight"

    elif class_name=="Potato___healthy":
        return "Healthy potato plant"

    elif class_name=="Tomato_Bacterial_spot":
        return "Tomato bacterial spot"

    elif class_name=="Tomato_Early_blight":
        return "Tomato early blight"

    elif class_name=="'Tomato_Late_blight'":
        return "Tomato late blight"

    elif class_name=="Tomato_Leaf_Mold":
        return "Tomato leaf mold"

    elif class_name=="Tomato_Septoria_leaf_spot":
        return "Tomato septoria leaf spot"

    elif class_name=="Tomato_Spider_mites_Two_spotted_spider_mite":
        return "Two-spotted spider mite"

    elif class_name=="Tomato__Target_Spot":
        return "Tomato target spot"

    elif class_name=="Tomato__Tomato_YellowLeaf__Curl_Virus":
        return "Tomato yellow leaf curl virus"

    elif class_name=="Tomato__Tomato_mosaic_virus":
        return "Tomato mosaic virus"

    elif class_name=="Tomato_healthy":
        return "Healthy tomato plant"
    
    else:
        return "Unknown"



def load_model():
    model,auto_transform=create_efficientnetb1_model(num_classes = 15, fine_tune = False)
    MODEL_PATH=os.path.join(PROJECT_ROOT,"experiments","PlantDiseaseIdentifier.pth") 
    model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device("cpu")))
    model.to("cpu")
    return model, auto_transform



TRAIN_DIR=os.path.join(PROJECT_ROOT,"data","PlantVillage","train")

CLASS_NAMES=['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___Late_blight',
 'Potato___healthy','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight',
 'Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']              # The class names list was obtained from the jupyter notebook

st.title("Plant Disease Identifier")
st.write("Upload an image of a leaf of your plant to identify the disease that it might has")
st.write(":green[**Note:** This app only works for bell pepper, potato and tomato plants]")
model, auto_transform = load_model()
uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    st.image(pil, caption="Input", use_container_width=True)
    inp = preprocess_image(pil,auto_transform)
    label, prob, all_probs = predict_image(model, inp, CLASS_NAMES, DEVICE)
    indexed_list=[(index,value) for index,value in enumerate(all_probs)]
    indexed_list.sort(key=lambda x: x[1], reverse=True)
    max_indeces=[index for index,value in indexed_list[:3]]         # 3 because we want to show the 3 highest probabilities
    st.write("## Disease probability:")
    for i in max_indeces:
        st.write(f":orange[- **{class_to_phrase(CLASS_NAMES[i])}:**]&nbsp;&nbsp;&nbsp;&nbsp;{all_probs[i]*100:.2f}%")
    st.write(":green[**Note:** Only the 3 most probable diseases are listed above.]")
