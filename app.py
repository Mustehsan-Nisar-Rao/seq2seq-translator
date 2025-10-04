import streamlit as st
import torch
import sentencepiece as spm
import os
import requests
import zipfile
from model import create_model

# =========================
# Configuration
# =========================
SP_MODEL_URL = "https://github.com/Mustehsan-Nisar-Rao/seq2seq-translator/blob/main/joint_char.model"
MODEL_WITH_ATTENTION_URL = "https://github.com/Mustehsan-Nisar-Rao/seq2seq-translator/releases/download/v.1/best_model.zip"
MODEL_WITHOUT_ATTENTION_URL = "https://github.com/Mustehsan-Nisar-Rao/seq2seq-translator/releases/download/v.1/best_seq2seq.pth"

SP_MODEL_PATH = "joint_char.model"
MODEL_WITH_ATTENTION_ZIP_PATH = "best_model.zip"
MODEL_WITH_ATTENTION_EXTRACTED_PATH = "best_model.pth"
MODEL_WITHOUT_ATTENTION_PATH = "best_seq2seq.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Roman Urdu Terminator Cleaner
# =========================
ROMAN_URDU_TERMINATORS = [
    "hai", "hain", "hoon", "ho",          # Present tense
    "tha", "thi", "thay",                 # Past tense
    "ga", "gi", "ge",                     # Future tense
    "chahiye",                            # Necessity
    "sakta hai", "sakti hai", "sakte hain"  # Ability
]

def clean_translation_roman(text: str) -> str:
    """Truncate translation at the last Roman Urdu terminator word."""
    text = text.strip().lower()
    for term in ROMAN_URDU_TERMINATORS:
        if term in text:
            return text[:text.rfind(term) + len(term)]
    return text

# =========================
# Download Models
# =========================
def download_model_with_attention():
    """Download ZIP file and extract the model with attention"""
    if os.path.exists(MODEL_WITH_ATTENTION_EXTRACTED_PATH):
        file_size = os.path.getsize(MODEL_WITH_ATTENTION_EXTRACTED_PATH)
        if file_size > 100000:
            st.success(f"✅ Attention model already available ({file_size:,} bytes)")
            return MODEL_WITH_ATTENTION_EXTRACTED_PATH
        else:
            st.warning("⚠️ Existing attention model file seems small, re-downloading...")
            os.remove(MODEL_WITH_ATTENTION_EXTRACTED_PATH)

    if not os.path.exists(MODEL_WITH_ATTENTION_ZIP_PATH):
        try:
            st.info("📥 Downloading attention model ZIP file...")
            response = requests.get(MODEL_WITH_ATTENTION_URL, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            progress_bar = st.progress(0)
            status_text = st.empty()

            with open(MODEL_WITH_ATTENTION_ZIP_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
                            status_text.text(f"Downloaded: {downloaded:,}/{total_size:,} bytes")

            progress_bar.empty()
            status_text.empty()

            final_size = os.path.getsize(MODEL_WITH_ATTENTION_ZIP_PATH)
            if total_size > 0 and final_size != total_size:
                st.error("❌ Attention model download incomplete")
                return None

            st.success(f"✅ Attention model ZIP downloaded successfully! ({final_size:,} bytes)")

        except Exception as e:
            st.error(f"❌ Attention model ZIP download failed: {e}")
            return None

    try:
        st.info("📦 Extracting attention model from ZIP...")
        with zipfile.ZipFile(MODEL_WITH_ATTENTION_ZIP_PATH, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            st.info(f"Files in ZIP: {file_list}")

            model_files = [f for f in file_list if f.endswith(('.pth', '.pt', '.bin'))]
            if not model_files:
                st.error("❌ No model file found in ZIP archive")
                return None

            model_file_in_zip = model_files[0]
            zip_ref.extract(model_file_in_zip)

            if model_file_in_zip != MODEL_WITH_ATTENTION_EXTRACTED_PATH:
                os.rename(model_file_in_zip, MODEL_WITH_ATTENTION_EXTRACTED_PATH)

            extracted_size = os.path.getsize(MODEL_WITH_ATTENTION_EXTRACTED_PATH)
            st.success(f"✅ Attention model extracted successfully! ({extracted_size:,} bytes)")
            return MODEL_WITH_ATTENTION_EXTRACTED_PATH

    except Exception as e:
        st.error(f"❌ Attention model ZIP extraction failed: {e}")
        return None

def download_model_without_attention():
    """Download the model without attention (direct .pth file)"""
    if os.path.exists(MODEL_WITHOUT_ATTENTION_PATH):
        file_size = os.path.getsize(MODEL_WITHOUT_ATTENTION_PATH)
        if file_size > 100000:
            st.success(f"✅ Non-attention model already available ({file_size:,} bytes)")
            return MODEL_WITHOUT_ATTENTION_PATH
        else:
            st.warning("⚠️ Existing non-attention model file seems small, re-downloading...")
            os.remove(MODEL_WITHOUT_ATTENTION_PATH)

    try:
        st.info("📥 Downloading non-attention model...")
        response = requests.get(MODEL_WITHOUT_ATTENTION_URL, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        progress_bar = st.progress(0)
        status_text = st.empty()

        with open(MODEL_WITHOUT_ATTENTION_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        status_text.text(f"Downloaded: {downloaded:,}/{total_size:,} bytes")

        progress_bar.empty()
        status_text.empty()

        final_size = os.path.getsize(MODEL_WITHOUT_ATTENTION_PATH)
        if total_size > 0 and final_size != total_size:
            st.error("❌ Non-attention model download incomplete")
            return None

        st.success(f"✅ Non-attention model downloaded successfully! ({final_size:,} bytes)")
        return MODEL_WITHOUT_ATTENTION_PATH

    except Exception as e:
        st.error(f"❌ Non-attention model download failed: {e}")
        return None

# =========================
# Load Tokenizer
# =========================
@st.cache_resource
def load_tokenizer():
    if not os.path.exists(SP_MODEL_PATH):
        try:
            st.info("📥 Downloading tokenizer...")
            response = requests.get(SP_MODEL_URL, timeout=30)
            response.raise_for_status()
            with open(SP_MODEL_PATH, 'wb') as f:
                f.write(response.content)
            st.success("✅ Tokenizer downloaded!")
        except Exception as e:
            st.error(f"❌ Tokenizer download failed: {e}")
            return None

    try:
        sp = spm.SentencePieceProcessor()
        sp.load(SP_MODEL_PATH)
        st.success("✅ Tokenizer loaded successfully!")
        return sp
    except Exception as e:
        st.error(f"❌ Error loading tokenizer: {e}")
        return None

# =========================
# Load Models
# =========================
@st.cache_resource
def load_models(_sp):
    if _sp is None:
        st.error("❌ Cannot load models: tokenizer not available")
        return None, None

    # Load model with attention
    model_with_attention_path = download_model_with_attention()
    if model_with_attention_path is None:
        st.warning("🔄 Using demonstration model for attention model (pretrained weights not available)")
        model_with_attention = create_test_model(_sp, "attention")
    else:
        try:
            VOCAB_SIZE = _sp.get_piece_size()
            st.info("🔄 Initializing attention model architecture...")
            model_with_attention = create_model(input_dim=VOCAB_SIZE, output_dim=VOCAB_SIZE, device=DEVICE)

            st.info("🔄 Loading attention model weights...")
            checkpoint = torch.load(model_with_attention_path, map_location=DEVICE, weights_only=False)

            if 'model_state_dict' in checkpoint:
                model_with_attention.load_state_dict(checkpoint['model_state_dict'])
                st.success("✅ Loaded attention model from model_state_dict")
            elif 'state_dict' in checkpoint:
                model_with_attention.load_state_dict(checkpoint['state_dict'])
                st.success("✅ Loaded attention model from state_dict")
            else:
                model_with_attention.load_state_dict(checkpoint)
                st.success("✅ Loaded attention model from weights file")

            model_with_attention.eval()
            model_with_attention.to(DEVICE)

            param_count_attention = sum(p.numel() for p in model_with_attention.parameters())
            st.success(f"🎉 Attention model loaded successfully! ({param_count_attention:,} parameters)")

        except Exception as e:
            st.error(f"❌ Attention model loading failed: {e}")
            st.warning("🔄 Falling back to demonstration model for attention...")
            model_with_attention = create_test_model(_sp, "attention")

    # Load model without attention
    model_without_attention_path = download_model_without_attention()
    if model_without_attention_path is None:
        st.warning("🔄 Using demonstration model for non-attention model (pretrained weights not available)")
        model_without_attention = create_test_model(_sp, "non_attention")
    else:
        try:
            VOCAB_SIZE = _sp.get_piece_size()
            st.info("🔄 Initializing non-attention model architecture...")
            # Note: You might need a different create_model function for non-attention
            # For now, using the same architecture but different weights
            model_without_attention = create_model(input_dim=VOCAB_SIZE, output_dim=VOCAB_SIZE, device=DEVICE)

            st.info("🔄 Loading non-attention model weights...")
            checkpoint = torch.load(model_without_attention_path, map_location=DEVICE, weights_only=False)

            if 'model_state_dict' in checkpoint:
                model_without_attention.load_state_dict(checkpoint['model_state_dict'])
                st.success("✅ Loaded non-attention model from model_state_dict")
            elif 'state_dict' in checkpoint:
                model_without_attention.load_state_dict(checkpoint['state_dict'])
                st.success("✅ Loaded non-attention model from state_dict")
            else:
                model_without_attention.load_state_dict(checkpoint)
                st.success("✅ Loaded non-attention model from weights file")

            model_without_attention.eval()
            model_without_attention.to(DEVICE)

            param_count_no_attention = sum(p.numel() for p in model_without_attention.parameters())
            st.success(f"🎉 Non-attention model loaded successfully! ({param_count_no_attention:,} parameters)")

        except Exception as e:
            st.error(f"❌ Non-attention model loading failed: {e}")
            st.warning("🔄 Falling back to demonstration model for non-attention...")
            model_without_attention = create_test_model(_sp, "non_attention")

    return model_with_attention, model_without_attention

def create_test_model(sp, model_type):
    if sp is None:
        return None
    VOCAB_SIZE = sp.get_piece_size()
    st.info(f"🔧 Creating demonstration {model_type} model...")
    model = create_model(
        input_dim=VOCAB_SIZE, output_dim=VOCAB_SIZE, device=DEVICE,
        enc_hid_dim=128, dec_hid_dim=128, emb_dim=64, enc_layers=1, dec_layers=2
    )
    model.eval()
    model.to(DEVICE)
    demo_param_count = sum(p.numel() for p in model.parameters())
    st.info(f"🔧 Demonstration {model_type} model created ({demo_param_count:,} parameters)")
    return model

# =========================
# Translation Functions
# =========================
def translate_with_attention(sentence, max_len=50):
    if sp is None or model_with_attention is None:
        return "Error: Attention model or tokenizer not loaded"
    try:
        tokens = sp.encode(sentence, out_type=int)
        src_tensor = torch.LongTensor([sp.bos_id()] + tokens + [sp.eos_id()]).to(DEVICE)
        src_tensor = src_tensor.unsqueeze(1)
        with torch.no_grad():
            encoder_outputs, hidden, cell = model_with_attention.encoder(src_tensor)
            hidden = hidden.unsqueeze(0).repeat(model_with_attention.decoder.rnn.num_layers, 1, 1)
            cell = cell.unsqueeze(0).repeat(model_with_attention.decoder.rnn.num_layers, 1, 1)
            input_tensor = torch.LongTensor([sp.bos_id()]).to(DEVICE)
            translated_tokens = []
            for i in range(max_len):
                output, hidden, cell = model_with_attention.decoder(input_tensor, hidden, cell, encoder_outputs)
                pred_token = output.argmax(1).item()
                if pred_token == sp.eos_id():
                    break
                translated_tokens.append(pred_token)
                input_tensor = torch.LongTensor([pred_token]).to(DEVICE)
            translated_text = sp.decode(translated_tokens) if translated_tokens else ""
            return clean_translation_roman(translated_text)
    except Exception as e:
        return f"Attention translation error: {e}"

def translate_without_attention(sentence, max_len=50):
    if sp is None or model_without_attention is None:
        return "Error: Non-attention model or tokenizer not loaded"
    try:
        tokens = sp.encode(sentence, out_type=int)
        src_tensor = torch.LongTensor([sp.bos_id()] + tokens + [sp.eos_id()]).to(DEVICE)
        src_tensor = src_tensor.unsqueeze(1)
        with torch.no_grad():
            encoder_outputs, hidden, cell = model_without_attention.encoder(src_tensor)
            hidden = hidden.unsqueeze(0).repeat(model_without_attention.decoder.rnn.num_layers, 1, 1)
            cell = cell.unsqueeze(0).repeat(model_without_attention.decoder.rnn.num_layers, 1, 1)
            input_tensor = torch.LongTensor([sp.bos_id()]).to(DEVICE)
            translated_tokens = []
            for i in range(max_len):
                output, hidden, cell = model_without_attention.decoder(input_tensor, hidden, cell, encoder_outputs)
                pred_token = output.argmax(1).item()
                if pred_token == sp.eos_id():
                    break
                translated_tokens.append(pred_token)
                input_tensor = torch.LongTensor([pred_token]).to(DEVICE)
            translated_text = sp.decode(translated_tokens) if translated_tokens else ""
            return clean_translation_roman(translated_text)
    except Exception as e:
        return f"Non-attention translation error: {e}"

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Seq2Seq Translator", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
st.title("🧠 Seq2Seq Neural Machine Translator - Compare Models")
st.markdown("---")

if 'resources_loaded' not in st.session_state:
    st.session_state.resources_loaded = False

st.sidebar.title("🔧 System Status")
if not st.session_state.resources_loaded:
    with st.sidebar:
        with st.spinner("🔄 Loading tokenizer..."):
            sp = load_tokenizer()
        with st.spinner("🔄 Loading models..."):
            model_with_attention, model_without_attention = load_models(sp)
    if sp is not None and model_with_attention is not None and model_without_attention is not None:
        st.session_state.resources_loaded = True
        st.session_state.sp = sp
        st.session_state.model_with_attention = model_with_attention
        st.session_state.model_without_attention = model_without_attention
    else:
        st.sidebar.error("❌ Failed to load resources")

sp = st.session_state.get('sp')
model_with_attention = st.session_state.get('model_with_attention')
model_without_attention = st.session_state.get('model_without_attention')

if sp and model_with_attention and model_without_attention:
    st.sidebar.success("✅ System Ready!")
    st.sidebar.info(f"**Vocabulary size:** {sp.get_piece_size()}")
    
    param_count_attention = sum(p.numel() for p in model_with_attention.parameters())
    param_count_no_attention = sum(p.numel() for p in model_without_attention.parameters())
    
    st.sidebar.info(f"**Attention model parameters:** {param_count_attention:,}")
    st.sidebar.info(f"**Non-attention model parameters:** {param_count_no_attention:,}")
    
    if param_count_attention < 1000000:
        st.sidebar.warning("⚠️ Using demonstration attention model")
    else:
        st.sidebar.success("✅ Using pretrained attention model!")
        
    if param_count_no_attention < 1000000:
        st.sidebar.warning("⚠️ Using demonstration non-attention model")
    else:
        st.sidebar.success("✅ Using pretrained non-attention model!")
else:
    st.sidebar.error("❌ System not ready")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("📥 Input Text")
    if sp is None or model_with_attention is None or model_without_attention is None:
        st.error("Please wait while the models load...")
    st.markdown("**Quick examples:**")
    example_cols = st.columns(3)
    examples = ["آج موسم بہت خوشگوار ہے", "کیا آپ میری مدد کر سکتے ہیں", "زندگی ایک خوبصورت سفر ہے"]
    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(example, key=f"ex_{i}"):
                if 'input_text' not in st.session_state:
                    st.session_state.input_text = ""
                st.session_state.input_text = example

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

user_input = st.text_area(
    "Enter text to translate:",
    height=120,
    placeholder="Type your text here...",
    key="input_text"
)

with col2:
    st.subheader("🔍 With Attention")
    attention_btn = st.button("🚀 Translate with Attention", type="primary", use_container_width=True, 
                             disabled=not (sp and model_with_attention))
    
    if attention_btn:
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to translate")
        else:
            with st.spinner("🔍 Translating with attention..."):
                translation_attention = translate_with_attention(user_input)
            if translation_attention.startswith("Attention translation error:"):
                st.error(f"❌ {translation_attention}")
            else:
                st.success("✅ Attention translation completed!")
                st.text_area("Translation with Attention:", translation_attention, height=120, key="attention_output")

with col3:
    st.subheader("⚡ Without Attention")
    no_attention_btn = st.button("🚀 Translate without Attention", type="secondary", use_container_width=True, 
                                disabled=not (sp and model_without_attention))
    
    if no_attention_btn:
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to translate")
        else:
            with st.spinner("🔍 Translating without attention..."):
                translation_no_attention = translate_without_attention(user_input)
            if translation_no_attention.startswith("Non-attention translation error:"):
                st.error(f"❌ {translation_no_attention}")
            else:
                st.success("✅ Non-attention translation completed!")
                st.text_area("Translation without Attention:", translation_no_attention, height=120, key="no_attention_output")

# Combined translation button
st.markdown("---")
compare_btn = st.button("🔄 Compare Both Models", type="primary", use_container_width=True,
                       disabled=not (sp and model_with_attention and model_without_attention))

if compare_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to translate")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            with st.spinner("🔍 Translating with attention..."):
                translation_attention = translate_with_attention(user_input)
            st.subheader("🔍 With Attention")
            if translation_attention.startswith("Attention translation error:"):
                st.error(f"❌ {translation_attention}")
            else:
                st.success("✅ Translation completed!")
                st.text_area("With Attention:", translation_attention, height=120, key="compare_attention")
        
        with col2:
            with st.spinner("🔍 Translating without attention..."):
                translation_no_attention = translate_without_attention(user_input)
            st.subheader("⚡ Without Attention")
            if translation_no_attention.startswith("Non-attention translation error:"):
                st.error(f"❌ {translation_no_attention}")
            else:
                st.success("✅ Translation completed!")
                st.text_area("Without Attention:", translation_no_attention, height=120, key="compare_no_attention")

with st.expander("📁 File Information"):
    st.write("**Current files:**")
    files_to_check = [
        (SP_MODEL_PATH, "Tokenizer"), 
        (MODEL_WITH_ATTENTION_ZIP_PATH, "Attention Model ZIP"), 
        (MODEL_WITH_ATTENTION_EXTRACTED_PATH, "Extracted Attention Model"),
        (MODEL_WITHOUT_ATTENTION_PATH, "Non-attention Model")
    ]
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            st.write(f"✅ **{description}:** {size:,} bytes")
        else:
            st.write(f"❌ **{description}:** Not found")
    if st.button("🔄 Clear Cache and Reload"):
        for file_path in [MODEL_WITH_ATTENTION_ZIP_PATH, MODEL_WITH_ATTENTION_EXTRACTED_PATH, MODEL_WITHOUT_ATTENTION_PATH]:
            if os.path.exists(file_path):
                os.remove(file_path)
        st.session_state.resources_loaded = False
        st.rerun()

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    Built with ❤️ using Streamlit • PyTorch • SentencePiece
</div>
<div style="text-align: center; color: gray;">
    Develop By: Mustehsan Nisar Rao
</div>
""", unsafe_allow_html=True)
