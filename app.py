import streamlit as st
import torch
import sentencepiece as spm
import os
import requests
import random
from model import create_model

# =========================
# Configuration
# =========================
SP_MODEL_URL = "https://github.com/Mustehsan-Nisar-Rao/seq2seq-translator/releases/download/v1/joint_char.model"
MODEL_URL = "https://github.com/Mustehsan-Nisar-Rao/seq2seq-translator/releases/download/v1/best_model.pth"

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
            total_size = int(r.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            
            with open(local_path, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(min(progress, 1.0))
            
            progress_bar.empty()
            st.success(f"{os.path.basename(local_path)} downloaded successfully!")
            return True
        except Exception as e:
            st.error(f"Failed to download {os.path.basename(local_path)}: {e}")
            return False
    return True

# =========================
# Load Tokenizer
# =========================
@st.cache_resource
def load_tokenizer(sp_model_path):
    if not download_file(SP_MODEL_URL, sp_model_path):
        return None
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(sp_model_path)
        st.success("✅ Tokenizer loaded successfully!")
        return sp
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None

# =========================
# Load Model (UPDATED WITH weights_only=False)
# =========================
@st.cache_resource
def load_model(_sp):
    if _sp is None:
        return None
        
    if not download_file(MODEL_URL, MODEL_PATH):
        return None
        
    try:
        INPUT_DIM = _sp.get_piece_size()
        OUTPUT_DIM = _sp.get_piece_size()
        
        st.info("🔄 Initializing model architecture...")
        model = create_model(INPUT_DIM, OUTPUT_DIM, DEVICE)
        
        st.info("🔄 Loading model weights...")
        
        # FIX: Added weights_only=False to handle PyTorch 2.6 compatibility
        try:
            # First try with weights_only=False (for PyTorch 2.6+)
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success("✅ Loaded from full checkpoint (model_state_dict)")
        else:
            model.load_state_dict(checkpoint)
            st.success("✅ Loaded from weights-only file")
            
        model.eval()
        model.to(DEVICE)
        st.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        
        # Try alternative loading methods
        st.info("🔄 Trying alternative loading methods...")
        
        try:
            # Method 2: Try with pickle module directly
            import pickle
            with open(MODEL_PATH, 'rb') as f:
                checkpoint = pickle.load(f)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.eval()
            model.to(DEVICE)
            st.success("✅ Model loaded successfully with alternative method!")
            return model
        except Exception as e2:
            st.error(f"Alternative loading failed: {e2}")
            
        try:
            # Method 3: Try loading with specific encoding
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, 
                                  pickle_module=pickle, 
                                  encoding='latin1')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.eval()
            model.to(DEVICE)
            st.success("✅ Model loaded with encoding workaround!")
            return model
        except Exception as e3:
            st.error(f"Encoding workaround failed: {e3}")
            
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

# Initialize session state for loading status
if 'resources_loaded' not in st.session_state:
    st.session_state.resources_loaded = False

if not st.session_state.resources_loaded:
    with st.sidebar:
        with st.spinner("Loading tokenizer..."):
            sp = load_tokenizer(SP_MODEL_PATH)
        
        with st.spinner("Loading model..."):
            model = load_model(sp)
    
    st.session_state.resources_loaded = True
    st.session_state.sp = sp
    st.session_state.model = model
else:
    sp = st.session_state.sp
    model = st.session_state.model

# Display model info in sidebar
if sp is not None and model is not None:
    st.sidebar.success("✅ All components loaded!")
    st.sidebar.info(f"**Vocabulary size:** {sp.get_piece_size()}")
    st.sidebar.info(f"**Device:** {DEVICE}")
    st.sidebar.info(f"**Model parameters:** {sum(p.numel() for p in model.parameters()):,}")
else:
    st.sidebar.error("❌ Failed to load components")
    
    # Add reload button if failed
    if st.sidebar.button("🔄 Retry Loading"):
        st.session_state.resources_loaded = False
        st.rerun()

# =========================
# Streamlit UI
# =========================
st.title("🧠 Seq2Seq Neural Machine Translator")
st.markdown("---")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Translation Settings")
    max_length = st.slider("Maximum translation length", 10, 100, 50)
    show_details = st.checkbox("Show token details", value=False)
    
    st.markdown("---")
    st.header("📊 Model Info")
    if sp is not None:
        st.write(f"**Special tokens:**")
        st.write(f"- PAD: {sp.pad_id()}")
        st.write(f"- SOS: {sp.bos_id()}") 
        st.write(f"- EOS: {sp.eos_id()}")
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("""
    This app uses a Seq2Seq model with attention for translation.
    - Character-level translation
    - Attention mechanism
    - Real-time inference
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Text")
    
    # Quick examples
    st.markdown("**Quick Examples:**")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    examples = ["Hello", "Thank you", "How are you?"]
    for i, example in enumerate(examples):
        with [example_col1, example_col2, example_col3][i]:
            if st.button(example, key=f"example_{i}"):
                if 'input_text' not in st.session_state:
                    st.session_state.input_text = ""
                st.session_state.input_text = example
    
    # Text input
    user_input = st.text_area(
        "Enter text to translate:", 
        height=150, 
        placeholder="Type your text here...",
        key="input_text" if 'input_text' in st.session_state else None
    )

with col2:
    st.subheader("📤 Translation Result")
    
    translate_button = st.button("🚀 Translate", type="primary", use_container_width=True)
    
    if translate_button:
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to translate.")
        elif sp is None or model is None:
            st.error("❌ Model or tokenizer failed to load. Please check the sidebar status.")
        else:
            with st.spinner("🔍 Translating..."):
                translation = translate_sentence(user_input, max_len=max_length)
            
            if translation.startswith("Translation error:"):
                st.error(f"❌ {translation}")
            else:
                st.success("✅ Translation completed!")
                st.text_area(
                    "Translation:", 
                    translation, 
                    height=150,
                    key="translation_output"
                )
                
                if show_details and translation:
                    try:
                        with st.expander("🔍 Token Details"):
                            input_tokens = sp.encode(user_input, out_type=str)
                            output_tokens = sp.encode(translation, out_type=str)
                            st.write(f"**Input tokens:** {input_tokens}")
                            st.write(f"**Output tokens:** {output_tokens}")
                            st.write(f"**Input length:** {len(input_tokens)} tokens")
                            st.write(f"**Output length:** {len(output_tokens)} tokens")
                    except Exception as e:
                        st.warning(f"Could not display token details: {e}")

# Debug information (collapsible)
with st.expander("🔧 Debug Information"):
    st.write(f"**Device:** {DEVICE}")
    st.write(f"**Tokenizer loaded:** {sp is not None}")
    st.write(f"**Model loaded:** {model is not None}")
    if sp is not None:
        st.write(f"**Vocabulary size:** {sp.get_piece_size()}")
    if model is not None:
        st.write(f"**Model parameters:** {sum(p.numel() for p in model.parameters()):,}")
    
    # Model architecture info
    if model is not None:
        st.write("**Model Architecture:**")
        st.code(f"""
        Encoder LSTM: {model.encoder.rnn.num_layers} layers
        Decoder LSTM: {model.decoder.rnn.num_layers} layers  
        Hidden dim: {model.encoder.rnn.hidden_size}
        Embedding dim: {model.encoder.embedding.embedding_dim}
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray; font-size:0.8em;">
Built with ❤️ using Streamlit • PyTorch • SentencePiece
</div>
""", unsafe_allow_html=True)
