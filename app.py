import streamlit as st
import torch
import sentencepiece as spm
import os
import requests
from model import create_model  # your model.py file

# =========================
# Configuration
# =========================
SP_MODEL_URL = "https://github.com/YourUsername/seq2seq-translator/releases/download/v1/joint_char.model"
MODEL_URL = "https://github.com/YourUsername/seq2seq-translator/releases/download/v1/best_model.pth"

SP_MODEL_PATH = "joint_char.model"
MODEL_PATH = "best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Helper: download file if missing
# =========================
def download_file(url, local_path):
    if not os.path.exists(local_path):
        try:
            st.info(f"Downloading {os.path.basename(local_path)}...")
            r = requests.get(url, stream=True)
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            st.success(f"{os.path.basename(local_path)} downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download {os.path.basename(local_path)}: {e}")

# =========================
# Load Tokenizer
# =========================
@st.cache_resource
def load_tokenizer(sp_model_path):
    download_file(SP_MODEL_URL, sp_model_path)
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    return sp

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model(_sp):
    download_file(MODEL_URL, MODEL_PATH)
    INPUT_DIM = _sp.get_piece_size()
    OUTPUT_DIM = _sp.get_piece_size()
    model = create_model(INPUT_DIM, OUTPUT_DIM, DEVICE)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(DEVICE)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# =========================
# Translation function
# =========================
def translate_sentence(sentence, max_len=50):
    if sp is None or model is None:
        return "Error: Model or tokenizer not loaded"
    try:
        tokens = sp.encode(sentence, out_type=int)
        src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            encoder_outputs, hidden, cell = model.encoder(src_tensor)
            hidden = hidden.unsqueeze(0).repeat(model.decoder.rnn.num_layers, 1, 1)
            cell = cell.unsqueeze(0).repeat(model.decoder.rnn.num_layers, 1, 1)

        trg_indexes = [sp.bos_id()]
        for i in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(DEVICE)
            with torch.no_grad():
                output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            if pred_token == sp.eos_id():
                break
        translated = sp.decode(trg_indexes[1:-1])
        return translated
    except Exception as e:
        return f"Translation error: {e}"

# =========================
# Load resources
# =========================
sp = load_tokenizer(SP_MODEL_PATH)
model = load_model(sp)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Seq2Seq Translator", page_icon="🚀", layout="wide")
st.title("🧠 Seq2Seq Translator")
st.markdown("---")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    max_length = st.slider("Maximum translation length", 10, 100, 50)
    show_details = st.checkbox("Show token details", value=False)
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("""
    This app uses a Seq2Seq model with attention for translation.
    - Character-level translation
    - Attention mechanism
    - Real-time inference
    """)

# Main content
col1, col2 = st.columns([1,1])
with col1:
    st.subheader("📥 Input Text")
    user_input = st.text_area("Enter text to translate:", height=150, placeholder="Type here...")
    
    # Quick examples
    st.markdown("**Quick Examples:**")
    example_col1, example_col2, example_col3 = st.columns(3)
    with example_col1:
        if st.button("Hello"):
            st.session_state.input_text = "Hello"
    with example_col2:
        if st.button("Thank you"):
            st.session_state.input_text = "Thank you"
    with example_col3:
        if st.button("How are you?"):
            st.session_state.input_text = "How are you?"

with col2:
    st.subheader("📤 Translation Result")
    if st.button("🚀 Translate", type="primary"):
        if not user_input.strip():
            st.warning("⚠️ Please enter text.")
        elif sp is None or model is None:
            st.error("❌ Model or tokenizer failed to load.")
        else:
            with st.spinner("Translating..."):
                translation = translate_sentence(user_input, max_len=max_length)
            st.success("✅ Translation completed!")
            st.text_area("Translation:", translation, height=150)
            if show_details:
                try:
                    input_tokens = sp.encode(user_input, out_type=str)
                    output_tokens = sp.encode(translation, out_type=str)
                    st.write(f"**Input tokens:** {input_tokens}")
                    st.write(f"**Output tokens:** {output_tokens}")
                except:
                    pass

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray; font-size:0.8em;">
Built with Streamlit • PyTorch • SentencePiece
</div>
""", unsafe_allow_html=True)
