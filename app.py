import streamlit as st
import torch
import sentencepiece as spm
import os
import requests
import hashlib
from model import create_model

# =========================
# Configuration
# =========================
SP_MODEL_URL = "https://github.com/YourUsername/seq2seq-translator/releases/download/v1/joint_char.model"
MODEL_URL = "https://github.com/YourUsername/seq2seq-translator/releases/download/v1/best_model.pth"

SP_MODEL_PATH = "joint_char.model"
MODEL_PATH = "best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Enhanced Download with Verification
# =========================
def download_file_with_retry(url, local_path, max_retries=3):
    """Download file with retry mechanism and verification"""
    for attempt in range(max_retries):
        try:
            if os.path.exists(local_path):
                st.info(f"✅ {os.path.basename(local_path)} already exists")
                return True
                
            st.info(f"📥 Downloading {os.path.basename(local_path)} (Attempt {attempt + 1}/{max_retries})...")
            
            # Get file size first
            head_response = requests.head(url, allow_redirects=True)
            file_size = int(head_response.headers.get('content-length', 0))
            
            # Download with progress
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Check if file is not empty
            if file_size == 0:
                st.warning(f"⚠️ File appears to be empty: {os.path.basename(local_path)}")
                if os.path.exists(local_path):
                    os.remove(local_path)
                continue
            
            downloaded_size = 0
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if file_size > 0:
                            progress = downloaded_size / file_size
                            progress_bar.progress(min(progress, 1.0))
                            status_text.text(f"Downloaded: {downloaded_size}/{file_size} bytes ({progress:.1%})")
            
            progress_bar.empty()
            status_text.empty()
            
            # Verify file size
            actual_size = os.path.getsize(local_path)
            if file_size > 0 and actual_size != file_size:
                st.warning(f"⚠️ File size mismatch: expected {file_size}, got {actual_size}")
                os.remove(local_path)
                continue
                
            # Basic file validation
            if actual_size == 0:
                st.warning("⚠️ Downloaded file is empty")
                os.remove(local_path)
                continue
                
            st.success(f"✅ {os.path.basename(local_path)} downloaded successfully! ({actual_size} bytes)")
            return True
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Download failed (attempt {attempt + 1}): {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            if attempt == max_retries - 1:
                st.error(f"💥 Failed to download {os.path.basename(local_path)} after {max_retries} attempts")
                return False
            st.info("🔄 Retrying in 3 seconds...")
            import time
            time.sleep(3)
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            return False
    
    return False

def verify_model_file(file_path):
    """Basic verification that the file is a valid PyTorch model"""
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist"
            
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "File is empty"
            
        if file_size < 1000:  # Model files should be reasonably large
            return False, f"File too small ({file_size} bytes), likely corrupted"
            
        # Try to read the first few bytes to check if it's a valid file
        with open(file_path, 'rb') as f:
            header = f.read(10)
            if len(header) < 10:
                return False, "File too short"
                
        return True, f"File looks valid ({file_size} bytes)"
        
    except Exception as e:
        return False, f"Verification failed: {e}"

# =========================
# Load Tokenizer
# =========================
@st.cache_resource
def load_tokenizer(sp_model_path):
    if not download_file_with_retry(SP_MODEL_URL, sp_model_path):
        return None
    try:
        # Verify tokenizer file
        is_valid, message = verify_model_file(sp_model_path)
        if not is_valid:
            st.error(f"❌ Tokenizer file invalid: {message}")
            return None
            
        sp = spm.SentencePieceProcessor()
        sp.load(sp_model_path)
        st.success("✅ Tokenizer loaded successfully!")
        return sp
    except Exception as e:
        st.error(f"❌ Error loading tokenizer: {e}")
        return None

# =========================
# Load Model with Enhanced Error Handling
# =========================
@st.cache_resource
def load_model(_sp):
    if _sp is None:
        st.error("❌ Cannot load model: tokenizer not available")
        return None
        
    if not download_file_with_retry(MODEL_URL, MODEL_PATH):
        return None
        
    try:
        # Verify model file before loading
        is_valid, message = verify_model_file(MODEL_PATH)
        if not is_valid:
            st.error(f"❌ Model file invalid: {message}")
            return None
            
        st.success(f"✅ Model file verified: {message}")
        
        INPUT_DIM = _sp.get_piece_size()
        OUTPUT_DIM = _sp.get_piece_size()
        
        st.info("🔄 Initializing model architecture...")
        model = create_model(INPUT_DIM, OUTPUT_DIM, DEVICE)
        
        st.info("🔄 Loading model weights...")
        
        # Try multiple loading strategies
        loading_strategies = [
            {"name": "weights_only=False", "kwargs": {"weights_only": False}},
            {"name": "default loading", "kwargs": {}},
            {"name": "pickle loading", "kwargs": {"pickle_module": __import__('pickle')}},
        ]
        
        checkpoint = None
        successful_strategy = None
        
        for strategy in loading_strategies:
            try:
                st.info(f"Trying {strategy['name']}...")
                checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, **strategy['kwargs'])
                successful_strategy = strategy['name']
                st.success(f"✅ Loaded with {strategy['name']}")
                break
            except Exception as e:
                st.warning(f"⚠️ {strategy['name']} failed: {str(e)[:100]}...")
                continue
        
        if checkpoint is None:
            st.error("💥 All loading strategies failed")
            return None
            
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success("✅ Loaded from full checkpoint")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            st.success("✅ Loaded from state_dict")
        else:
            # Assume it's a direct state dict
            model.load_state_dict(checkpoint)
            st.success("✅ Loaded from weights-only file")
            
        model.eval()
        model.to(DEVICE)
        st.success(f"✅ Model loaded successfully using {successful_strategy}!")
        return model
        
    except Exception as e:
        st.error(f"💥 Final model loading failed: {e}")
        
        # Provide detailed troubleshooting info
        st.error("""
        **Troubleshooting steps:**
        1. Check if the model file URL is correct
        2. Ensure the model file is not corrupted
        3. Try re-uploading the model file
        4. Check if the model architecture matches the saved weights
        """)
        
        return None

# =========================
# Translation function
# =========================
def translate_sentence(sentence, max_len=50):
    if sp is None or model is None:
        return "Error: Model or tokenizer not loaded"
    
    try:
        # Tokenize the input sentence
        tokens = sp.encode(sentence, out_type=int)
        
        # Add SOS and EOS tokens
        src_tensor = torch.LongTensor([sp.bos_id()] + tokens + [sp.eos_id()]).to(DEVICE)
        src_tensor = src_tensor.unsqueeze(1)  # Add batch dimension [seq_len, 1]
        
        with torch.no_grad():
            # Get encoder outputs
            encoder_outputs, hidden, cell = model.encoder(src_tensor)
            
            # Prepare hidden and cell states for decoder
            hidden = hidden.unsqueeze(0).repeat(model.decoder.rnn.num_layers, 1, 1)
            cell = cell.unsqueeze(0).repeat(model.decoder.rnn.num_layers, 1, 1)
            
            # Start with SOS token
            input_tensor = torch.LongTensor([sp.bos_id()]).to(DEVICE)
            
            translated_tokens = []
            
            for i in range(max_len):
                output, hidden, cell = model.decoder(input_tensor, hidden, cell, encoder_outputs)
                
                # Get the predicted token
                pred_token = output.argmax(1).item()
                
                # Stop if EOS token is generated
                if pred_token == sp.eos_id():
                    break
                
                translated_tokens.append(pred_token)
                input_tensor = torch.LongTensor([pred_token]).to(DEVICE)
        
        # Decode the translated tokens
        if translated_tokens:
            translated_text = sp.decode(translated_tokens)
        else:
            translated_text = ""
            
        return translated_text
        
    except Exception as e:
        return f"Translation error: {e}"

# =========================
# Initialize App
# =========================
st.set_page_config(
    page_title="Seq2Seq Translator", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Load resources
# =========================
st.sidebar.title("🔧 Model Status")

# Initialize session state
if 'resources_loaded' not in st.session_state:
    st.session_state.resources_loaded = False
    st.session_state.load_attempts = 0

if not st.session_state.resources_loaded and st.session_state.load_attempts < 3:
    with st.sidebar:
        st.info(f"🔄 Loading attempt {st.session_state.load_attempts + 1}/3")
        
        with st.spinner("Loading tokenizer..."):
            sp = load_tokenizer(SP_MODEL_PATH)
        
        with st.spinner("Loading model..."):
            model = load_model(sp)
    
    st.session_state.load_attempts += 1
    
    if sp is not None and model is not None:
        st.session_state.resources_loaded = True
        st.session_state.sp = sp
        st.session_state.model = model
    else:
        st.session_state.resources_loaded = False
else:
    sp = st.session_state.get('sp', None)
    model = st.session_state.get('model', None)

# Display status
if sp is not None and model is not None:
    st.sidebar.success("✅ All components loaded!")
    st.sidebar.info(f"**Vocabulary size:** {sp.get_piece_size()}")
    st.sidebar.info(f"**Device:** {DEVICE}")
else:
    st.sidebar.error("❌ Failed to load components")
    
    if st.session_state.load_attempts >= 3:
        st.sidebar.error("💥 Maximum load attempts reached")
        
    # Manual reload button
    if st.sidebar.button("🔄 Force Reload"):
        st.session_state.resources_loaded = False
        st.session_state.load_attempts = 0
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        if os.path.exists(SP_MODEL_PATH):
            os.remove(SP_MODEL_PATH)
        st.rerun()

# =========================
# Streamlit UI
# =========================
st.title("🧠 Seq2Seq Neural Machine Translator")
st.markdown("---")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Text")
    
    if sp is None or model is None:
        st.warning("⚠️ Model not loaded. Check the sidebar status.")
    
    # Quick examples
    st.markdown("**Quick Examples:**")
    examples = ["Hello", "Thank you", "How are you?"]
    cols = st.columns(3)
    for i, example in enumerate(examples):
        with cols[i]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.input_text = example
    
    user_input = st.text_area(
        "Enter text to translate:", 
        height=150,
        value=st.session_state.get('input_text', ''),
        key="user_input"
    )

with col2:
    st.subheader("📤 Translation Result")
    
    translate_button = st.button("🚀 Translate", type="primary", use_container_width=True,
                               disabled=sp is None or model is None)
    
    if translate_button and user_input.strip():
        with st.spinner("🔍 Translating..."):
            translation = translate_sentence(user_input, max_len=50)
        
        if translation.startswith("Translation error:"):
            st.error(f"❌ {translation}")
        else:
            st.success("✅ Translation completed!")
            st.text_area("Translation:", translation, height=150, key="translation_output")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray; font-size:0.8em;">
Built with ❤️ using Streamlit • PyTorch • SentencePiece
</div>
""", unsafe_allow_html=True)
