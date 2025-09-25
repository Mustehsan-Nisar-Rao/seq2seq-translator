import streamlit as st
import torch
import sentencepiece as spm
from model import create_model  # your Seq2Seq model

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Seq2Seq Translator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Load Tokenizer with Caching
# =========================
@st.cache_resource
def load_tokenizer():
    try:
        SP_MODEL = "joint_char.model"  # make sure this file exists
        sp = spm.SentencePieceProcessor()
        sp.load(SP_MODEL)
        return sp
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None

# =========================
# Load Model with Caching (state_dict)
# =========================
@st.cache_resource
def load_model(_sp):
    try:
        MODEL_PATH = "best_model.pth"  # should contain 'model_state_dict'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        INPUT_DIM = _sp.get_piece_size()
        OUTPUT_DIM = _sp.get_piece_size()

        model = create_model(INPUT_DIM, OUTPUT_DIM, device)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# =========================
# Load resources
# =========================
sp = load_tokenizer()
if sp is not None:
    model, device = load_model(sp)
else:
    model, device = None, None

# =========================
# Inference function
# =========================
def translate_sentence(sentence, max_len=50):
    if sp is None or model is None:
        return "Error: Model or tokenizer not loaded properly"

    try:
        tokens = sp.encode(sentence, out_type=int)
        src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)

        with torch.no_grad():
            encoder_outputs, hidden, cell = model.encoder(src_tensor)
            hidden = hidden.unsqueeze(0).repeat(model.decoder.rnn.num_layers, 1, 1)
            cell = cell.unsqueeze(0).repeat(model.decoder.rnn.num_layers, 1, 1)

        trg_indexes = [sp.bos_id()]
        for i in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
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
st.title("🧠 Seq2Seq Translator")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Settings")
    max_length = st.slider("Maximum translation length", 10, 100, 50)
    show_details = st.checkbox("Show translation details", value=False)
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("""
    This app uses a Sequence-to-Sequence model with attention for translation.
    - Character-level translation
    - Attention mechanism
    - Real-time inference
    """)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Text")
    user_input = st.text_area(
        "Enter text to translate:",
        placeholder="Type your text here...",
        height=150,
        key="input_text"
    )
    
    # Quick examples
    st.markdown("**Quick examples:**")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        if st.button("Hello"):
            st.session_state.input_text = "Hello"
    with ex2:
        if st.button("Thank you"):
            st.session_state.input_text = "Thank you"
    with ex3:
        if st.button("How are you?"):
            st.session_state.input_text = "How are you?"

with col2:
    st.subheader("📤 Translation Result")
    if st.button("🚀 Translate", type="primary"):
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
                st.text_area("Translation:", translation, height=150, key="output_text")
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

st.markdown("---")
st.markdown("""
<style>
.footer {
    text-align: center;
    color: gray;
    font-size: 0.8em;
}
</style>
<div class="footer">
    Built with Streamlit • PyTorch • SentencePiece
</div>
""", unsafe_allow_html=True)
