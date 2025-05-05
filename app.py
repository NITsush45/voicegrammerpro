import os
os.environ["STREAMLIT_WATCH_DISABLE"] = "true"

import streamlit as st
import time
import whisper
import sys
from audio_utils import convert_to_wav
from transcriber import transcribe_audio
from scorer import score_grammar
import nest_asyncio


os.makedirs('.streamlit', exist_ok=True)
with open('.streamlit/config.toml', 'w') as f:
    f.write("""
[server]
headless = true
port = $PORT
enableCORS = false
enableXsrfProtection = false
address = "0.0.0.0"
""")

model = whisper.load_model("base", device="cpu")
sys.modules["torch.classes"] = None
nest_asyncio.apply()

st.set_page_config(
    page_title="VoiceGrammar Pro",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    h1, h2, h3 {
        color: #2E4057;
        font-weight: 600;
    }
    
    .animate-header {
        animation: fadeInDown 1s ease-out;
    }
    
    .stButton>button {
        background-color: #048BA8;
        color: white;
        font-weight: 500;
        padding: 10px 24px;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        background-color: #0396A6;
    }
    
    .file-uploader {
        border: 2px dashed #ccc;
        border-radius: 15px;
        padding: 30px 20px;
        text-align: center;
        background-color: #f8f9fa;
        transition: all 0.3s ease;
        animation: fadeIn 1s ease-out;
    }
    
    .file-uploader:hover {
        border-color: #048BA8;
        background-color: #f0f7fa;
    }
    
    .score-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-top: 20px;
        animation: slideInUp 0.6s ease-out;
    }
    
    .score-circle {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        background: conic-gradient(#048BA8 0%, #048BA8 var(--score-percent), #f0f0f0 var(--score-percent), #f0f0f0 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        position: relative;
    }
    
    .score-circle::before {
        content: "";
        width: 130px;
        height: 130px;
        background-color: white;
        border-radius: 50%;
        position: absolute;
    }
    
    .score-value {
        position: relative;
        font-size: 36px;
        font-weight: 700;
        color: #2E4057;
        z-index: 1;
    }
    
    .issue-card {
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 12px 15px;
        margin: 8px 0;
        border-radius: 4px;
        animation: fadeIn 0.8s ease-out;
    }
    
    .success-card {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 12px 15px;
        border-radius: 4px;
        animation: fadeIn 0.8s ease-out;
    }
    
    .transcript-box {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-top: 20px;
        animation: slideInUp 0.8s ease-out;
    }
    
    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 30px;
    }
    
    .accent-text {
        color: #048BA8;
        font-weight: 500;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #4caf50;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(76, 175, 80, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
        }
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/voice-presentation.png", width=80)
    st.markdown("<h2 class='animate-header'>VoiceGrammar Pro</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown("""
    1. Upload your audio recording  
    2. Our AI transcribes your speech  
    3. Grammar analysis is performed  
    4. Receive detailed feedback
    """)
    st.markdown("---")
    st.markdown("### Supported Formats")
    st.markdown("• WAV\n• MP3\n• M4A")
    st.markdown("---")

    demo_button = st.button("Try Demo Recording")

    # Dataset sample selection
    st.markdown("### Try Samples from Dataset")
    dataset_type = st.radio("Choose Dataset", ["Train", "Test"])
    sample_folder = "train" if dataset_type == "Train" else "test"
    sample_files = [f for f in os.listdir(sample_folder) if f.endswith((".wav", ".mp3", ".m4a"))]
    selected_sample = st.selectbox("Select a Sample", sample_files)
    load_sample_button = st.button("Load Selected Sample")

st.markdown("<h1 class='animate-header'>Grammar Scoring Engine</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload your voice recording to analyze grammar and pronunciation quality.</p>", unsafe_allow_html=True)

st.markdown("<div class='file-uploader'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["wav", "mp3", "m4a"])
if not uploaded_file:
    st.markdown("Drag and drop your audio file here or click to browse", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None or demo_button or load_sample_button:
    os.makedirs("temp", exist_ok=True)

    if demo_button and not uploaded_file and not load_sample_button:
        input_path = "demo_audio/hello.mp3"
        st.info("Using demo recording...")

    elif load_sample_button:
        input_path = os.path.join(sample_folder, selected_sample)
        st.info(f"Using {dataset_type} sample: {selected_sample}")

    else:
        input_path = f"temp/{uploaded_file.name}"
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

    st.audio(input_path)

    wav_path = os.path.splitext(input_path)[0] + ".wav"
    convert_to_wav(input_path, wav_path)

    with st.spinner():
        st.markdown("""
        <div class='loading-animation'>
            <div>
                <div class='status-indicator'></div>
                <span>Processing audio...</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        progress_bar = st.progress(0)
        for i in range(100):
            if i < 30:
                status_text = "Analyzing audio waveform..."
            elif i < 60:
                status_text = "Transcribing speech..."
            else:
                status_text = "Evaluating grammar..."

            progress_bar.progress(i + 1)
            if i % 10 == 0:
                time.sleep(0.05)

        text = transcribe_audio(wav_path)
        score, issues = score_grammar(text)

    progress_bar.empty()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<div class='transcript-box'>", unsafe_allow_html=True)
        st.markdown("<h3>Transcription</h3>", unsafe_allow_html=True)
        st.write(text)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='score-card'>", unsafe_allow_html=True)
        st.markdown("<h3>Grammar Score</h3>", unsafe_allow_html=True)
        score_percent = score if score <= 100 else 100
        st.markdown(f"""
        <div style="text-align: center; margin: 20px 0;">
            <div class="score-circle" style="--score-percent: {score_percent}%; ">
                <div class="score-value">{score}</div>
            </div>
            <p style="margin-top: 10px; font-size: 18px;">
                <span class="accent-text">
                    {"Excellent" if score >= 90 else "Good" if score >= 70 else "Fair" if score >= 50 else "Needs Improvement"}
                </span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='score-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Feedback & Suggestions</h3>", unsafe_allow_html=True)
    if issues:
        for issue in issues:
            st.markdown(f"""
            <div class='issue-card' style='color: black;'>
                <strong>{issue.get("ruleIssueType", "Unknown Rule")}:</strong> {issue.get("message", "No message provided")}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='success-card'>
            <strong>Perfect!</strong> No grammar issues were found in your speech. Great job!
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='score-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Improvement Tips</h3>", unsafe_allow_html=True)
    if score >= 90:
        st.markdown("""
        <p>Your grammar is excellent! To maintain this level:</p>
        <ul>
            <li>Continue practicing with more complex speech patterns</li>
            <li>Try different topics to expand your vocabulary</li>
            <li>Consider recording formal presentations for additional practice</li>
        </ul>
        """, unsafe_allow_html=True)
    elif score >= 70:
        st.markdown("""
        <p>You have good grammar skills. To improve further:</p>
        <ul>
            <li>Focus on the specific errors highlighted above</li>
            <li>Practice speaking more slowly to reduce mistakes</li>
            <li>Try reading aloud from professionally written texts</li>
        </ul>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <p>Here are some tips to improve your grammar:</p>
        <ul>
            <li>Practice the specific issues mentioned above</li>
            <li>Try shorter sentences to maintain proper structure</li>
            <li>Listen to native speakers and imitate their patterns</li>
            <li>Consider using grammar practice apps daily</li>
        </ul>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    timestamp = time.strftime("%Y-%m-%d %H:%M")
    st.session_state.history.append({
        "timestamp": timestamp,
        "score": score,
        "text": text[:50] + "..." if len(text) > 50 else text
    })

    # Clean up uploaded file (not for demo or dataset sample)
    if not demo_button and not load_sample_button:
        os.remove(input_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

# Show history
if "history" in st.session_state and st.session_state.history:
    with st.expander("Previous Analyses"):
        for idx, entry in enumerate(reversed(st.session_state.history)):
            st.markdown(f"""
            <div style="padding: 10px 0; border-bottom: 1px solid #eee;">
                <strong>{entry['timestamp']}</strong> - Score: {entry['score']}/100<br>
                <small>{entry['text']}</small>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 15px; background: linear-gradient(90deg, #f8f9fa, #e9ecef, #f8f9fa); border-radius: 10px; margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" class="footer-container">
    <p style="color: #495057; font-size: 16px; margin: 5px 0; animation: fadeInUp 1.5s ease-out, pulse 3s infinite alternate;">
        <span style="animation: colorShift 5s infinite;">©</span> 2025 
        <span style="font-weight: bold; animation: glow 2s ease-in-out infinite;">VoiceGrammar Pro</span> 
        <span style="margin: 0 5px;">|</span> 
        <span style="animation: slideIn 2s ease-out;">AI-Powered Grammar Analysis Tool</span>
    </p>
</div>

<style>
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    from { transform: scale(1); }
    to { transform: scale(1.03); }
}

@keyframes colorShift {
    0% { color: #495057; }
    25% { color: #0066cc; }
    50% { color: #339933; }
    75% { color: #993399; }
    100% { color: #495057; }
}

@keyframes glow {
    0% { text-shadow: 0 0 5px rgba(0,102,204,0.3); }
    50% { text-shadow: 0 0 15px rgba(0,102,204,0.7); }
    100% { text-shadow: 0 0 5px rgba(0,102,204,0.3); }
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}
</style>
""", unsafe_allow_html=True)
