import streamlit as st
import torch
import sentencepiece as spm
from model import create_model  # import your model.py
import os
import requests

# =========================
# Configuration
# =========================
MODEL_FILENAME = "best_model.pth"
MODEL_URL = "https://github.com/Mustehsan-Nisar-Rao/seq2seq-translator/releases/download/v1/best_model.pth"  # change to your actual release URL
SP_MODEL_FILENAME = "joint_char.model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Helper functions
# =========================
def download_file(url, local_path):
    """Download file from URL if it does not exist."""
    if not os.path.exists(local_path):
        st.info(f"Downloading {local_path} ...")
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        chunk_size = 1024
        with open(local_path, 'wb') as f:
            for data in r.iter_content(chunk_size):
                f.write(data)
        st.success(f"{local_path} downloaded successfully!")

# =========================
# Load Tokenizer
# =========================
@st.cache_resource
def load_tokenizer():
    try:
        if not os.path.exists(SP_MODEL_FILENAME):
            st.error(f"Tokenizer file '{SP_MODEL_FILENAME}' not found!")
            return None
        sp = spm.SentencePieceProcessor()
        sp.load(SP_MODEL_FILENAME)
        return sp
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model(_sp):
    try:
        download_file(MODEL_URL, MODEL_FILENAME)
        INPUT_DIM = _sp.get_piece_size()
        OUTPUT_DIM = _sp.get_piece_size()
        model = create_model(INPUT_DIM, OUTPUT_DIM, DEVICE)
        state_dict = torch.load(MODEL_FILENAME, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(DEVICE)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# =========================
# Translation Function
# =========================
def translate_sentence(sentence, max_len=50):
    if sp is None or model is None:
        return "Error: Model or tokenizer not loaded properly"
    
    try:
        tokens = sp.encode(sentence, out_type=int)
        src_tensor = torch.LongTensor([sp.bos_id()] + tokens + [sp.eos_id()]).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            encoder_outputs, hidden, cell = model.encoder(src_tensor)
            hidden = hidden.unsqueeze(0).repeat(model.decoder.rnn.num_layers, 1, 1)
            cell = cell.unsqueeze(0).repeat(model.decoder.rnn.num_layers, 1, 1)

        trg_indexes = [sp.bos_id()]
        for _ in range(max_len):
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
        return f"Translation error: {str(e)}"

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Seq2Seq Translator", page_icon="🚀", layout="wide")

st.title("🧠 Seq2Seq Translator")
st.markdown("---")

# Load resources
sp = load_tokenizer()
model = load_model(sp) if sp else None

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_length = st.slider("Maximum translation length", 10, 100, 50)
    show_details = st.checkbox("Show translation details", value=False)
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("""
    This app uses a Sequence-to-Sequence model with attention for translation.
    
    **Features:**
    - Character-level translation
    - Attention mechanism
    - Real-time inference
    """)

# Input and output
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Text")
    user_input = st.text_area("Enter text to translate:", height=150)
    
with col2:
    st.subheader("📤 Translation Result")
    if st.button("🚀 Translate"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to translate.")
        elif sp is None or model is None:
            st.error("❌ Model or tokenizer failed to load. Please check the files.")
        else:
            with st.spinner("Translating..."):
                translation = translate_sentence(user_input, max_len=max_length)
            
            if translation.startswith("Error:"):
                st.error(translation)
            else:
                st.success("✅ Translation completed!")
                st.text_area("Translation:", translation, height=150)
                
                if show_details:
                    with st.expander("🔍 Translation Details"):
                        st.write(f"**Input length:** {len(user_input)} characters")
                        st.write(f"**Output length:** {len(translation)} characters")
                        try:
                            input_tokens = sp.encode(user_input, out_type=str)
                            output_tokens = sp.encode(translation, out_type=str)
                            st.write(f"**Input tokens:** {input_tokens}")
                            st.write(f"**Output tokens:** {output_tokens}")
                        except:
                            pass

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;color:gray;font-size:0.8em;'>
        Built with Streamlit • PyTorch • SentencePiece
    </div>
    """,
    unsafe_allow_html=True
)
