
import streamlit as st
import time
import os 
import boto3
from transformers import pipeline
import torch
from PIL import Image

# Set page config first
st.set_page_config(
    page_title="Quick ML App",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    /* Main container with subtle gradient */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
        padding: 2rem;
    }
    
    /* Typography hierarchy with smooth transitions */
    h1 {
        color: #2c3e50;
        font-weight: 700;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #3498db, #2c3e50) 1;
        padding-bottom: 12px;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    h2 {
        color: #2980b9;
        font-weight: 600;
        margin-top: 2rem;
        position: relative;
        padding-left: 1rem;
    }
    
    h2:before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 4px;
        background: linear-gradient(to bottom, #3498db, #2c3e50);
        border-radius: 4px;
    }
    
    /* Modern card design with hover effects */
    .card {
        background: white;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        padding: 2rem;
        margin-bottom: 2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.12);
    }
    
    /* Buttons with modern gradient and micro-interactions */
    .stButton>button {
        background: linear-gradient(135deg, #6e8efb 0%, #4a6cf7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 4px 8px rgba(74, 108, 247, 0.25) !important;
        letter-spacing: 0.5px;
    }
            
    /* TARGET PRIMARY BUTTONS SPECIFICALLY */
    button[kind="primary"] {
        background: linear-gradient(135deg, #6e8efb 0%, #4a6cf7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 8px rgba(74, 108, 247, 0.25) !important;
    }
    /* HOVER STATES */
    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 16px rgba(74, 108, 247, 0.3) !important;
    }
    /* DISABLED STATES */
    button[kind="primary"]:disabled {
        opacity: 0.7 !important;
        background: linear-gradient(135deg, #b0c0f0 0%, #8a9ef0 100%) !important;
    }
            
    /* Specifically target disabled buttons with help tooltips */
    .stButton>button:disabled {
        background: linear-gradient(135deg, #6e8efb 0%, #4a6cf7 100%) !important;
        opacity: 0.7 !important;
        color: white !important;
    }

            
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 16px rgba(74, 108, 247, 0.3) !important;
    }
    
    .stButton>button:active {
        transform: translateY(0) !important;
        box-shadow: 0 4px 8px rgba(74, 108, 247, 0.3) !important;
    }
            
    /* Target buttons with help tooltips */
    .stButton button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #6e8efb 0%, #4a6cf7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 4px 8px rgba(74, 108, 247, 0.25) !important;
        letter-spacing: 0.5px;
    }

    /* Hover states for both */
    .stButton>button:hover,
    .stButton button[data-testid="baseButton-secondary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 16px rgba(74, 108, 247, 0.3) !important;
    }
    
    /* Input fields with modern styling */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stNumberInput>div>div>input {
        border-radius: 12px !important;
        border: 1px solid #dfe6e9 !important;
        padding: 12px 16px !important;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
        transition: border 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus,
    .stNumberInput>div>div>input:focus {
        border-color: #4a6cf7 !important;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05), 0 0 0 2px rgba(74, 108, 247, 0.2);
    }
    
    /* Enhanced alerts and info boxes */
    .stAlert {
        border-radius: 12px !important;
        padding: 16px 20px !important;
    }
    
    .stInfo {
        background-color: #f0f7ff !important;
        border-left: 4px solid #4a6cf7 !important;
    }
    
    .stWarning {
        background-color: #fff8e6 !important;
        border-left: 4px solid #ffb74d !important;
    }
    
    .stSuccess {
        background-color: #edf7ed !important;
        border-left: 4px solid #66bb6a !important;
    }
    
    /* File uploader with modern look */
    .stFileUploader>div {
        border: 2px dashed #4a6cf7 !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        background: rgba(74, 108, 247, 0.03) !important;
        transition: all 0.3s ease;
    }
    
    .stFileUploader>div:hover {
        background: rgba(74, 108, 247, 0.08) !important;
    }
    
    /* Progress bar with gradient */
    .stProgress>div>div>div {
        background: linear-gradient(90deg, #6e8efb 0%, #4a6cf7 100%) !important;
        border-radius: 8px !important;
    }
    
    /* Tabs with modern styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 8px 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
        color: #64748b !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6e8efb 0%, #4a6cf7 100%) !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Result boxes with polished animations */
    .result-box {
        border-radius: 16px;
        padding: 24px;
        margin: 24px 0;
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .positive {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #66bb6a;
    }
    
    .negative {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 5px solid #ef5350;
    }
    
    /* Spinner with brand color */
    .stSpinner>div {
        border-color: #4a6cf7 transparent transparent transparent !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 16px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
        padding: 16px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.12) !important;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_PATH_Sentiment = "tinybert_sentiment_model"
S3_BUCKET_Sentiment = "udemy-sentiment-analysis"
S3_PREFIX_Sentiment = "model_folder/tinybert_sentiment_model/"

MODEL_PATH_Motion = "vit_human_pose_classifier"
S3_BUCKET_Motion = "udemy-human-pose-analysis"
S3_PREFIX_Motion = "model_folder/vit_human_pose_classifier/"

# Initialize session state
if 'sentiment_model_loaded' not in st.session_state:
    st.session_state.sentiment_model_loaded = False
if 'sentiment_model_downloaded' not in st.session_state:
    st.session_state.sentiment_model_downloaded = os.path.exists(MODEL_PATH_Sentiment)
if 'motion_model_loaded' not in st.session_state:
    st.session_state.motion_model_loaded = False
if 'motion_model_downloaded' not in st.session_state:
    st.session_state.motion_model_downloaded = os.path.exists(MODEL_PATH_Motion)
if 'uploaded_image_key' not in st.session_state:
    st.session_state.uploaded_image_key = 0

# S3 download function
def download_model_sentiment():
    s3 = boto3.client('s3')
    if not os.path.exists(MODEL_PATH_Sentiment):
        os.makedirs(MODEL_PATH_Sentiment)
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        paginator = s3.get_paginator('list_objects_v2')
        results = paginator.paginate(Bucket=S3_BUCKET_Sentiment, Prefix=S3_PREFIX_Sentiment)
        objects = []
        for result in results:
            if 'Contents' in result:
                objects.extend(result['Contents'])
        
        total_objects = len(objects)
        if total_objects == 0:
            st.error("No objects found in S3 bucket")
            return False
        
        for i, obj in enumerate(objects):
            object_name = obj['Key']
            file_path = os.path.join(MODEL_PATH_Sentiment, os.path.relpath(object_name, S3_PREFIX_Sentiment))
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            s3.download_file(S3_BUCKET_Sentiment, object_name, file_path)
            progress_bar.progress((i + 1) / total_objects)
            status_text.text(f"Downloading files: {i+1}/{total_objects}")
        
        return True
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return False

def download_model_motion():
    s3 = boto3.client('s3')
    if not os.path.exists(MODEL_PATH_Motion):
        os.makedirs(MODEL_PATH_Motion)
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        paginator = s3.get_paginator('list_objects_v2')
        results = paginator.paginate(Bucket=S3_BUCKET_Motion, Prefix=S3_PREFIX_Motion)
        objects = []
        for result in results:
            if 'Contents' in result:
                objects.extend(result['Contents'])
        
        total_objects = len(objects)
        if total_objects == 0:
            st.error("No objects found in S3 bucket")
            return False
        
        for i, obj in enumerate(objects):
            object_name = obj['Key']
            file_path = os.path.join(MODEL_PATH_Motion, os.path.relpath(object_name, S3_PREFIX_Motion))
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            s3.download_file(S3_BUCKET_Motion, object_name, file_path)
            progress_bar.progress((i + 1) / total_objects)
            status_text.text(f"Downloading files: {i+1}/{total_objects}")
        
        return True
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return False


# App layout
st.title("📊 Sentiment Analysis & Motion Detection App")
st.markdown("""
1. **Text Analysis**: Enter your text to analyze sentiment  
2. **Motion Detection**: Upload an image to classify human activity
3. **Model Management**: Download and load models as needed before use
""")
st.divider()

# Sidebar for model management
with st.sidebar:
    st.header("⚙️ Model Management")
    st.divider()

    # Sentiment Model Management
    st.subheader("Sentiment Model")
    if not st.session_state.sentiment_model_downloaded:
        st.warning("Sentiment Model not downloaded")
        if st.button("Download Sentiment Model", key="download_sentiment_button"):
            with st.spinner("Downloading sentiment model from S3..."):
                if download_model_sentiment():
                    st.session_state.sentiment_model_downloaded = True
                    st.success("Sentiment Model downloaded successfully!")
                    st.rerun()
    else:
        st.success("Sentiment Model already downloaded")
        
        if not st.session_state.sentiment_model_loaded:
            if st.button("Load Sentiment Model into Memory", key="load_sentiment_button"):
                with st.spinner("Loading Sentiment model..."):
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    try:
                        st.session_state.classifier = pipeline(
                            "text-classification",
                            model=MODEL_PATH_Sentiment,
                            device=device
                        )
                        st.session_state.sentiment_model_loaded = True
                        st.success("Sentiment Model loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
        else:
            st.success("Sentiment Model loaded and ready")
            if st.button("Unload Sentiment Model", key="unload_sentiment_button"):
                if 'classifier' in st.session_state:
                    del st.session_state.classifier
                st.session_state.sentiment_model_loaded = False
                st.info("Sentiment Model unloaded from memory")
                st.rerun()

    st.divider()

    # Vision Transformer Model Management
    st.subheader("Motion Detection Model")
    if not st.session_state.motion_model_downloaded:
        st.warning("Motion Detection Model not downloaded")
        if st.button("Download Motion Model", key="download_motion_button"):
            with st.spinner("Downloading motion model from S3..."):
                if download_model_motion():
                    st.session_state.motion_model_downloaded = True
                    st.success("Motion Model downloaded successfully!")
                    st.rerun()
    else:
        st.success("Motion Model already downloaded")
        
        if not st.session_state.motion_model_loaded:
            if st.button("Load Motion Model into Memory", key="load_motion_button"):
                with st.spinner("Loading Motion Detection model..."):
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    try:
                        st.session_state.motion_classifier = pipeline(
                            "image-classification",
                            model=MODEL_PATH_Motion,
                            device=device
                        )
                        st.session_state.motion_model_loaded = True
                        st.success("Motion Model loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
        else:
            st.success("Motion Model loaded and ready")
            if st.button("Unload Motion Model", key="unload_motion_button"):
                if 'motion_classifier' in st.session_state:
                    del st.session_state.motion_classifier
                st.session_state.motion_model_loaded = False
                st.info("Motion Model unloaded from memory")
                st.rerun()

# ------------------------------------------------------------------------
# Main content area
st.subheader("🔍 Text Analysis")

# Initialize or update text_input in session state
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

# Bind text_area to session state
text_input = st.text_area(
    "Enter your text here:",
    value=st.session_state.text_input,
    placeholder="Type something to analyze sentiment...",
    height=150
)

# Update session state when text changes
if text_input != st.session_state.text_input:
    st.session_state.text_input = text_input

col1, col2, col3 = st.columns([1.1, 0.9, 2])

with col1:
    predict_btn = st.button("Analyze Sentiment", 
                            disabled=not st.session_state.sentiment_model_loaded,
                            help="Please load the Sentiment Model first",
                            type="primary"
                            )
with col2:
    clear_btn = st.button(
        "Clear Text",
        key="clear_text_btn",
        help="Clear the text input box",
        type="primary"
        )

# Clear text when button is clicked
if clear_btn:
    st.session_state.text_input = ""
    st.rerun()  # Refresh to show empty box

if predict_btn and text_input.strip():
    with st.spinner("Analyzing sentiment..."):
        try:
            result = st.session_state.classifier(text_input)[0]
            
            # Display results
            st.subheader("📊 Analysis Results")
            
            emoji = "😊" if result['label'].upper() == 'POSITIVE' else "😞"
            confidence = result['score']
            sentiment_class = "positive" if result['label'].upper() == 'POSITIVE' else "negative"
            
            st.markdown(f"""
            <div class="result-box {sentiment_class}">
                <h3>Sentiment: {result['label']} {emoji}</h3>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence visualization
            st.progress(float(confidence), text=f"Confidence: {confidence:.2%}")
            
            # Interpretation
            if sentiment_class == "positive":
                st.info("This text appears to express **positive** sentiment")
            else:
                st.info("This text appears to express **negative** sentiment")
                
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
elif predict_btn and not text_input.strip():
    st.warning("Please enter some text to analyze")


st.divider()
# load the vit model for human motion detection

# ------------------------------------------------------------------------
st.subheader("🤖 Human Motion Detection")

# Motion Detection Section
uploaded_image = st.file_uploader(
    "Upload an image for motion detection:",
    type=["jpg", "jpeg", "png"],
    help="Upload an image containing a person to detect their motion/activity",
    key=f"file_uploader_{st.session_state.uploaded_image_key}"  
)

# Create columns for buttons
motion_col1, motion_col2, motion_col3 = st.columns([1, 1, 2])

with motion_col1:
    detect_btn = st.button(
        "Analyze Motion",
        disabled=not st.session_state.motion_model_loaded,
        help="Please load the Motion Detection Model first",
        type="primary"
    )

with motion_col2:
    delete_btn = st.button(
        "Delete Image",
        key="delete_image_btn",
        help="Remove the uploaded image",
        type="primary"
    )

if delete_btn:
    # Clear the uploaded image from session state
    st.session_state.uploaded_image_key += 1
    st.rerun()

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if detect_btn:
        with st.spinner("Detecting human motion..."):
            try:
                # Perform prediction
                results = st.session_state.motion_classifier(image)
                
                # Display results
                st.subheader("🏃‍♂️ Motion Detection Results")
                
                # Get top prediction
                top_result = results[0]
                label = top_result['label'].replace('_', ' ').title()
                confidence = top_result['score']
                
                # Define emojis for different motions
                motion_emojis = {
                    'running': '🏃‍♂️',
                    'listening to music': '🎧',
                    'walking': '🚶‍♂️',
                    'jumping': '🦘',
                    'standing': '🧍‍♂️',
                    'sitting': '🧘‍♂️',
                    'dancing': '💃',
                    'exercising': '🏋️‍♂️',
                    'laughing': '😂',
                    'crying': '😢' ,
                    'talking': '🗣️',
                    'eating': '🍽️',
                    'sleeping': '😴',
                    'gaming': '🎮',
                    'swimming': '🏊‍♂️',
                    'reading': '📖',
                    'cooking': '👨‍🍳',
                    'smiling': '😊',
                    'dancing': '💃',
                    'cycling': '🚴‍♂️',
                }
                
                # Get emoji or use default if not found
                emoji = motion_emojis.get(label.lower(), '❓')
                
                # Display the top result with styling
                st.markdown(f"""
                <div class="result-box">
                    <h3>Detected Motion: {label} {emoji}</h3>
                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence visualization
                st.progress(float(confidence), text=f"Confidence: {confidence:.2%}")
                
                # Show all predictions in an expandable section
                with st.expander("View detailed predictions"):
                    st.write("All detected motions with confidence scores:")
                    for result in results:
                        formatted_label = result['label'].replace('_', ' ').title()
                        st.write(f"- {formatted_label}: {result['score']:.2%}")
                
                # Add some fun interpretation based on the top result
                if 'running' in label.lower():
                    st.success("Great cardio workout detected! 🏃‍♂️💨")
                elif 'listening to music' in label.lower():
                    st.info("Enjoying some tunes! 🎶 Remember to stay aware of your surroundings.")
                elif 'walking' in label.lower():
                    st.success("A healthy walk! Perfect for daily activity. 🚶‍♂️")
                elif 'standing' in label.lower():
                    st.warning("Standing detected. Consider taking a stretch break if you've been standing long!")
                elif 'sitting' in label.lower():
                    st.warning("Sitting detected. Remember to stand up and move around periodically!")
                
            except Exception as e:
                st.error(f"Error during motion detection: {str(e)}")
    elif detect_btn and not uploaded_image:
        st.warning("Please upload an image first")




# Add some explanations
st.divider()
with st.expander("ℹ️ About this App"):
    st.markdown("""
    This app uses a fine-tuned TinyBERT model for sentiment analysis. Here's how it works:
    - Download the model from AWS S3 (first time only)
    - Load the model into memory
    - Enter text and click "Analyze Sentiment"
    - View the sentiment results (Positive/Negative) with confidence score
    
    The app also includes a Vision Transformer (ViT) model for human motion detection:
    - Analyzes the pose and context of people in images
    - Returns multiple possible activities with confidence scores
    
    **Note:** The first model load might take 30-60 seconds depending on your system.
    """)

st.caption("© 2025 First ML App Deployed with Streamlit and FastAPI")







