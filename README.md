
# Content Moderation System Using LLMs and VLMs

## Project Description
This project implements a content moderation system that utilizes both Language Models (LLMs) and Vision-Language Models (VLMs) to analyze and evaluate text and image content for harmfulness.

The system features:
- **Text Moderation**: Using the `cardiffnlp/twitter-roberta-base-offensive` model to classify text as harmful or safe, providing confidence scores for each classification.
- **Image Captioning and Moderation**: Leveraging the BLIP (Bootstrapped Language-Image Pre-training) model to generate captions for uploaded images and evaluate the captions for harmful content.
- **User-Friendly Interface**: Developed with Streamlit, the application allows users to input text and upload images easily.

## Features
- **Text Input**: Users can enter text to check for harmful content, receiving a detailed moderation result based on model confidence scores.
- **Image Upload**: Users can upload images, which are processed to generate captions and evaluate for harmful content.
- **Confidence Scoring**: The system provides clear confidence scores for both text and image moderation results, helping users understand the moderation output.

## Installation Instructions
To set up the project locally, clone the repository and install the required packages:

```bash
git clone git@github.com:haroon423/HarmSafe-Detection.git
cd HarmSafe-Detection
pip install -r requirements.txt
streamlit run app.py

