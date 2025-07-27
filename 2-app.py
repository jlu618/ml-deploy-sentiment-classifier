
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
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>   
    /* General body styling */
    section.main > div {
        max-width: 80rem !important;  
        padding-left: 1rem !important;  
        padding-right: 1rem !important;
    }             
            
    /* Main button styling */
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #1565C0;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:disabled {
        background-color: #90CAF9;
        cursor: not-allowed;
    }
    
    /* Secondary button styling */
    .secondary-button>button {
        background-color: white;
        color: #1E88E5;
        border: 1px solid #1E88E5;
    }
    .secondary-button>button:hover {
        background-color: #E3F2FD;
    }
    
    /* Text area styling */
    .stTextArea>div>div>textarea {
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #BBDEFB;
    }
    
    /* Header styling */
    .header {
        color: #0D47A1;
    }
    /* Result boxes */
    .result-box {
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(30, 136, 229, 0.1);
        border-left: 5px solid;
    }
    .positive {
        background-color: #E8F5E9;
        border-left-color: #43A047;
    }
    .negative {
        background-color: #FFEBEE;
        border-left-color: #E53935;
    }
    
    /* Progress spinner */
    .stSpinner>div {
        border-color: #1E88E5 transparent transparent transparent;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #E3F2FD;
    }
    .sidebar-header {
        color: #0D47A1;
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
st.markdown("Analyze text sentiment using TinyBERT model", unsafe_allow_html=True)
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
    predict_btn = st.button("Analyze Sentiment", disabled=not st.session_state.sentiment_model_loaded)
with col2:
    clear_btn = st.button("Clear Text")

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
    help="Upload an image containing a person to detect their motion/activity"
)

# Create columns for buttons
motion_col1, motion_col2, motion_col3 = st.columns([1, 1, 2])

with motion_col1:
    detect_btn = st.button(
        "Analyze Motion",
        disabled=not st.session_state.motion_model_loaded
    )

with motion_col2:
    delete_btn = st.button("Delete Image")

if delete_btn:
    # Clear the uploaded image from session state
    st.session_state.motion_image_uploader = None
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
                    'dancing': '💃'
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







