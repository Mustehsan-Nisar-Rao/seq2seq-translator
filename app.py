import streamlit as st
import torch
import sentencepiece as spm
from model import create_model
import requests
import os

# =========================
# Config
# =========================
MODEL_URL = "https://github.com/Mustehsan-Nisar-Rao/seq2seq-translator/releases/download/v1/best_model.pth"
MODEL_PATH = "best_model.pth"
SP_MODEL_PATH = "joint_char.model"

# =========================
# Download model if not exists
# =========================
def download_file(url, local_path):
    if not os.path.exists(local_path):
        with st.spinner(f"Downloading {local_path}..."):
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success(f"Downloaded {local_path}")

# =========================
# Load Tokenizer
# =========================
@st.cache_resource
def load_tokenizer(sp_model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    return sp

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model(model_path, sp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = sp.get_piece_size()
    output_dim = sp.get_piece_size()
    model = create_model(input_dim, output_dim, device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    model.to(device)
    return model, device

# =========================
# Main logic
# =========================
download_file(MODEL_URL, MODEL_PATH)
sp = load_tokenizer(SP_MODEL_PATH)
model, device = load_model(MODEL_PATH, sp)

st.success("Model and tokenizer loaded successfully ✅")
