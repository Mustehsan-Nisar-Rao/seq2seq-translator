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

MODEL_ZIP_URL = "https://github.com/Mustehsan-Nisar-Rao/seq2seq-translator/releases/download/v.1/best_model.zip"

SP_MODEL_PATH = "joint_char.model"
MODEL_ZIP_PATH = "best_model.zip"
MODEL_EXTRACTED_PATH = "best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Download and Extract ZIP
# =========================
def download_and_extract_model():
    """Download ZIP file and extract the model"""
    
    # Check if model is already extracted and valid
    if os.path.exists(MODEL_EXTRACTED_PATH):
        file_size = os.path.getsize(MODEL_EXTRACTED_PATH)
        if file_size > 100000:  # Model should be >100KB
            st.success(f"✅ Model already available ({file_size:,} bytes)")
            return MODEL_EXTRACTED_PATH
        else:
            st.warning("⚠️ Existing model file seems small, re-downloading...")
            os.remove(MODEL_EXTRACTED_PATH)
    
    # Download ZIP file if needed
    if not os.path.exists(MODEL_ZIP_PATH):
        try:
            st.info("📥 Downloading model ZIP file...")
            response = requests.get(MODEL_ZIP_URL, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with open(MODEL_ZIP_PATH, 'wb') as f:
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
            
            # Verify download
            final_size = os.path.getsize(MODEL_ZIP_PATH)
            if total_size > 0 and final_size != total_size:
                st.error("❌ Download incomplete")
                return None
                
            st.success(f"✅ ZIP downloaded successfully! ({final_size:,} bytes)")
            
        except Exception as e:
            st.error(f"❌ ZIP download failed: {e}")
            return None
    
    # Extract the ZIP file
    try:
        st.info("📦 Extracting model from ZIP...")
        
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            # List files in ZIP for debugging
            file_list = zip_ref.namelist()
            st.info(f"Files in ZIP: {file_list}")
            
            # Look for model files
            model_files = [f for f in file_list if f.endswith(('.pth', '.pt', '.bin'))]
            
            if not model_files:
                st.error("❌ No model file found in ZIP archive")
                return None
            
            # Extract the model file
            model_file_in_zip = model_files[0]
            zip_ref.extract(model_file_in_zip)
            
            # Rename to standard name if needed
            if model_file_in_zip != MODEL_EXTRACTED_PATH:
                os.rename(model_file_in_zip, MODEL_EXTRACTED_PATH)
            
            extracted_size = os.path.getsize(MODEL_EXTRACTED_PATH)
            st.success(f"✅ Model extracted successfully! ({extracted_size:,} bytes)")
            return MODEL_EXTRACTED_PATH
            
    except Exception as e:
        st.error(f"❌ ZIP extraction failed: {e}")
        return None

# =========================
# Load Tokenizer
# =========================
@st.cache_resource
def load_tokenizer():
    """Load the sentencepiece tokenizer"""
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
# Load Model
# =========================
@st.cache_resource
def load_model(_sp):
    """Load the seq2seq model with weights"""
    if _sp is None:
        st.error("❌ Cannot load model: tokenizer not available")
        return None
    
    # Download and extract model
    model_path = download_and_extract_model()
    
    if model_path is None:
        st.warning("🔄 Using demonstration model (pretrained weights not available)")
        return create_test_model(_sp)
    
    try:
        VOCAB_SIZE = _sp.get_piece_size()
        
        st.info("🔄 Initializing model architecture...")
        model = create_model(
            input_dim=VOCAB_SIZE,
            output_dim=VOCAB_SIZE,
            device=DEVICE
        )
        
        st.info("🔄 Loading model weights...")
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        
        # Load weights into model
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success("✅ Loaded from model_state_dict")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            st.success("✅ Loaded from state_dict")
        else:
            # Assume it's a direct state dict
            model.load_state_dict(checkpoint)
            st.success("✅ Loaded from weights file")
        
        model.eval()
        model.to(DEVICE)
        
        # Verify model loaded correctly
        param_count = sum(p.numel() for p in model.parameters())
        st.success(f"🎉 Model loaded successfully! ({param_count:,} parameters)")
        
        return model
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.warning("🔄 Falling back to demonstration model...")
        return create_test_model(_sp)

def create_test_model(sp):
    """Create a minimal model for demonstration purposes"""
    if sp is None:
        return None
        
    VOCAB_SIZE = sp.get_piece_size()
    
    st.info("🔧 Creating demonstration model...")
    
    model = create_model(
        input_dim=VOCAB_SIZE,
        output_dim=VOCAB_SIZE,
        device=DEVICE,
        enc_hid_dim=128,  # Smaller for demo
        dec_hid_dim=128,
        emb_dim=64,
        enc_layers=1,
        dec_layers=2
    )
    
    model.eval()
    model.to(DEVICE)
    
    demo_param_count = sum(p.numel() for p in model.parameters())
    st.info(f"🔧 Demonstration model created ({demo_param_count:,} parameters)")
    
    return model

# =========================
# Translation Function
# =========================
def translate_sentence(sentence, max_len=50):
    """Translate a single sentence"""
    if sp is None or model is None:
        return "Error: Model or tokenizer not loaded"
    
    try:
        # Tokenize input
        tokens = sp.encode(sentence, out_type=int)
        
        # Create source tensor with SOS and EOS
        src_tensor = torch.LongTensor([sp.bos_id()] + tokens + [sp.eos_id()]).to(DEVICE)
        src_tensor = src_tensor.unsqueeze(1)  # Add batch dimension [seq_len, 1]
        
        with torch.no_grad():
            # Encode the source
            encoder_outputs, hidden, cell = model.encoder(src_tensor)
            
            # Prepare decoder initial states
            hidden = hidden.unsqueeze(0).repeat(model.decoder.rnn.num_layers, 1, 1)
            cell = cell.unsqueeze(0).repeat(model.decoder.rnn.num_layers, 1, 1)
            
            # Start with SOS token
            input_tensor = torch.LongTensor([sp.bos_id()]).to(DEVICE)
            translated_tokens = []
            
            # Generate translation token by token
            for i in range(max_len):
                output, hidden, cell = model.decoder(input_tensor, hidden, cell, encoder_outputs)
                pred_token = output.argmax(1).item()
                
                # Stop if EOS token is generated
                if pred_token == sp.eos_id():
                    break
                
                translated_tokens.append(pred_token)
                input_tensor = torch.LongTensor([pred_token]).to(DEVICE)
        
        # Decode the tokens to text
            translated_text = sp.decode(translated_tokens) if translated_tokens else ""
            return translated_text
        
    except Exception as e:
        return f"Translation error: {e}"

# =========================
# Streamlit App
# =========================
st.set_page_config(
    page_title="Seq2Seq Translator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Seq2Seq Neural Machine Translator")
st.markdown("---")

# =========================
# Load Resources
# =========================
if 'resources_loaded' not in st.session_state:
    st.session_state.resources_loaded = False

# Sidebar for status
st.sidebar.title("🔧 System Status")

if not st.session_state.resources_loaded:
    with st.sidebar:
        with st.spinner("🔄 Loading tokenizer..."):
            sp = load_tokenizer()
        
        with st.spinner("🔄 Loading model..."):
            model = load_model(sp)
    
    if sp is not None and model is not None:
        st.session_state.resources_loaded = True
        st.session_state.sp = sp
        st.session_state.model = model
    else:
        st.sidebar.error("❌ Failed to load resources")

# Get resources from session state
sp = st.session_state.get('sp')
model = st.session_state.get('model')

# Display status in sidebar
if sp and model:
    st.sidebar.success("✅ System Ready!")
    st.sidebar.info(f"**Vocabulary size:** {sp.get_piece_size()}")
    
    param_count = sum(p.numel() for p in model.parameters())
    st.sidebar.info(f"**Model parameters:** {param_count:,}")
    
    if param_count < 1000000:
        st.sidebar.warning("⚠️ Using demonstration model")
    else:
        st.sidebar.success("✅ Using pretrained model!")
else:
    st.sidebar.error("❌ System not ready")

# =========================
# Main Translation Interface
# =========================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Text")
    
    if sp is None or model is None:
        st.error("Please wait while the model loads...")
    
    # Quick examples
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
    key="input_text"   # bind directly to session_state
)


with col2:
    st.subheader("📤 Translation Result")
    
    translate_btn = st.button(
        "🚀 Translate", 
        type="primary", 
        use_container_width=True,
        disabled=not (sp and model)
    )
    
    if translate_btn:
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to translate")
        else:
            with st.spinner("🔍 Translating..."):
                translation = translate_sentence(user_input)
            
            if translation.startswith("Translation error:"):
                st.error(f"❌ {translation}")
            else:
                st.success("✅ Translation completed!")
                st.text_area(
                    "Translation:",
                    translation,
                    height=120,
                    key="translation_output"
                )

# =========================
# File Information
# =========================
with st.expander("📁 File Information"):
    st.write("**Current files:**")
    
    files_to_check = [
        (SP_MODEL_PATH, "Tokenizer"),
        (MODEL_ZIP_PATH, "Model ZIP"),
        (MODEL_EXTRACTED_PATH, "Extracted Model")
    ]
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            st.write(f"✅ **{description}:** {size:,} bytes")
        else:
            st.write(f"❌ **{description}:** Not found")
    
    # Clear cache button
    if st.button("🔄 Clear Cache and Reload"):
        for file_path in [MODEL_ZIP_PATH, MODEL_EXTRACTED_PATH]:
            if os.path.exists(file_path):
                os.remove(file_path)
        st.session_state.resources_loaded = False
        st.rerun()

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    Built with ❤️ using Streamlit • PyTorch • SentencePiece
</div>
""", unsafe_allow_html=True)
