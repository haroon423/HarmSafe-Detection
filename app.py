import streamlit as st
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import torch

# Load text moderation model
@st.cache_resource
def load_text_moderation_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading text moderation model: {e}")
        return None, None

# Load CLIP model and processor for image analysis
@st.cache_resource
def load_clip_model():
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        return model, processor
    except Exception as e:
        st.error(f"Error loading CLIP model: {e}")
        return None, None

# Moderate text based on model output probabilities
def moderate_text(prompt, tokenizer, model, threshold=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)
    
    safe_score = probabilities[0][0].item()  # Assuming 0th index is 'safe'
    harmful_score = probabilities[0][1].item()  # Assuming 1st index is 'harmful'
    
    if harmful_score > safe_score:
        if harmful_score > threshold:
            return f"This content is harmful with {harmful_score * 100:.2f}% confidence."
        else:
            return f"Uncertain: this content might be harmful with {harmful_score * 100:.2f}% confidence."
    else:
        if safe_score > threshold:
            return f"This content is safe with {safe_score * 100:.2f}% confidence."
        else:
            return f"Uncertain: this content might be safe with {safe_score * 100:.2f}% confidence."

# Moderate image using CLIP
def moderate_image(image, processor, model):
    inputs = processor(text=["safe", "harmful"], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # This gives you the logits for image classification
    probabilities = torch.softmax(logits_per_image, dim=1)
    
    safe_score = probabilities[0][0].item()
    harmful_score = probabilities[0][1].item()
    
    if harmful_score > safe_score:
        return f"This image is harmful with {harmful_score * 100:.2f}% confidence."
    else:
        return f"This image is safe with {safe_score * 100:.2f}% confidence."

# Streamlit UI for Content Moderation
st.title("Content Moderation: Text and Image Analysis")

# Text Moderation Section
st.write("Enter text to check for harmful content.")
user_text = st.text_input("Enter text:")

if user_text:
    tokenizer, model = load_text_moderation_model()
    if tokenizer and model:
        with st.spinner("Checking text for harmful content..."):
            moderation_result = moderate_text(user_text, tokenizer, model)
            st.write("Moderation Result:")
            st.write(moderation_result)

# Image Moderation Section
st.write("Upload an image to check for harmful content.")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    clip_model, clip_processor = load_clip_model()
    if clip_model and clip_processor:
        with st.spinner("Checking image for harmful content..."):
            image_moderation_result = moderate_image(image, clip_processor, clip_model)
            st.write("Image Moderation Result:")
            st.write(image_moderation_result)
