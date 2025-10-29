import streamlit as st
import os
import torch
from transformers import pipeline
import boto3

# --- Configuration ---
BUCKET_NAME = 'bucket-for-practice-vineet'
S3_PREFIX = 'ml-models/tinybert-sentiment-analysis/'
LOCAL_MODEL_PATH = 'tinybert-sentiment-analysis'

# --- S3 Download Function (FIXED) ---
def download_dir(local_path, s3_prefix):
    """
    Downloads a directory from S3, creating local subdirectories.
    """
    s3 = boto3.client('s3')
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    
    for result in paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_prefix):
        if 'Contents' not in result:
            continue
            
        for key in result['Contents']:
            s3_key = key['Key']
            
            # Skip the "directory" itself
            if s3_key.endswith('/'):
                continue
                
            local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
            
            # --- THIS IS THE CRITICAL FIX ---
            # Ensure the local directory exists before downloading the file
            local_dir = os.path.dirname(local_file)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir, exist_ok=True)
            # ---------------------------------
            
            try:
                s3.download_file(BUCKET_NAME, s3_key, local_file)
            except Exception as e:
                st.error(f"Error downloading {s3_key}: {e}")

# --- Model Loading Function (CACHED) ---
@st.cache_resource
def load_model():
    """
    Loads the pipeline model only once and caches it.
    Specifies the local path.
    """
    st.write("Loading model...") # So you can see when this runs
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    classifier = pipeline('text-classification', model=LOCAL_MODEL_PATH, device=device)
    return classifier

# --- Streamlit App UI ---
st.title("ML Model Deployment At Server")

# Check if the model directory exists and is not empty
model_is_downloaded = os.path.isdir(LOCAL_MODEL_PATH) and len(os.listdir(LOCAL_MODEL_PATH)) > 0

if not model_is_downloaded:
    st.warning("Model files not found. Please download the model to proceed.")
    button = st.button("Download Model")
    if button:
        with st.spinner("Downloading... Please wait! This may take a few minutes."):
            download_dir(LOCAL_MODEL_PATH, S3_PREFIX)
            st.success("Model downloaded successfully!")
            st.info("Rerunning the app to load the model...")
            # Rerun the script to pass the 'if model_is_downloaded' check
            st.rerun() 
else:
    # --- Only show this if the model is downloaded ---
    st.success("Model is downloaded and ready!")

    # Load the model (this will be cached)
    classifier = load_model()

    text = st.text_area("Enter Your Review", "Type...")
    predict = st.button("Predict")

    if predict:
        if text and text != "Type...":
            with st.spinner("Predicting..."):
                output = classifier(text)
                st.write(output)
        else:
            st.warning("Please enter a review to predict.")