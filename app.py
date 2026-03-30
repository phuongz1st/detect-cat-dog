import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="AI Pet Detector", page_icon="✨", layout="centered")

# 2. Inject Custom CSS for modern redesign (2025 AI/SaaS style)
custom_css = """
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    /* Modern Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }

    /* Centered Card Layout */
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 4rem 3rem !important;
        border-radius: 24px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        max-width: 800px;
        margin-top: 3rem;
        margin-bottom: 3rem;
    }

    /* Typography */
    h1 {
        background: -webkit-linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 700 !important;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 16px;
        font-weight: 400;
        margin-bottom: 2.5rem;
    }
    
    .author-text {
        text-align: center;
        font-size: 12px;
        color: #94a3b8;
        margin-top: -1.5rem;
        margin-bottom: 3rem;
    }

    /* Modern File Uploader with Hover Effects */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 16px !important;
        background-color: #f8fafc;
        transition: all 0.3s ease;
        padding: 3rem;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #667eea !important;
        background-color: #eff6ff;
        transform: translateY(-2px);
    }
    
    /* Styled Upload Button inside uploader */
    [data-testid="stFileUploadDropzone"] button {
        background: white;
        color: #1e293b;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        font-weight: 500;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"] button:hover {
        background: #f1f5f9;
        color: #667eea;
        border-color: #667eea;
    }

    /* Modern Progress Bar */
    .stProgress > div > div > div {
        background-color: #667eea;
        border-radius: 10px;
    }
    
    /* Pre-Prediction Loading Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .loading-text {
        animation: pulse 1.5s infinite;
        font-weight: 500;
        color: #764ba2;
        text-align: center;
    }

    /* Modern Primary Button with Hover Effects */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.8rem 3rem;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.3);
        width: 100%;
        margin-top: 2rem;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(118, 75, 162, 0.4);
        color: white;
    }

    /* Styled Image Preview */
    [data-testid="stImage"] img {
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        transition: all 0.3s ease;
        margin-bottom: 2rem;
    }
    [data-testid="stImage"] img:hover {
        transform: scale(1.02);
    }

    /* Modern Result Badges with Category Styling and Bounce Animation */
    @keyframes fadeInBounce {
        0% { opacity: 0; transform: scale(0.8) translateY(20px); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }
    
    .result-badge {
        display: block;
        padding: 20px 30px;
        border-radius: 16px;
        font-size: 26px;
        font-weight: 700;
        text-align: center;
        margin-top: 3rem;
        margin-bottom: 2rem;
        animation: fadeInBounce 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        box-shadow: 0 8px 25px rgba(0,0,0,0.06);
        color: white;
    }
    
    /* Specific styling for Cat, Dog, and Unknown results */
    .badge-cat { background: linear-gradient(135deg, #f06292 0%, #c2185b 100%); }
    .badge-dog { background: linear-gradient(135deg, #66bb6a 0%, #388e3c 100%); }
    .badge-unknown { background: linear-gradient(135deg, #78909c 0%, #455a64 100%); }

    /* Prediction details list below the main badge */
    .prediction-details-title {
        font-weight: 600;
        font-size: 18px;
        color: #1e293b;
        margin-top: 2rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .other-predictions-list {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
        max-width: 600px;
        margin: 0 auto;
    }
    
    .other-pred-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #475569;
        font-size: 14px;
        padding: 8px 12px;
        border-radius: 8px;
        transition: background-color 0.2s;
    }
    .other-pred-item:hover {
        background-color: #f1f5f9;
    }
    .other-pred-label { font-weight: 500; }
    .other-pred-category { color: #94a3b8; }
    .other-pred-prob { font-weight: 600; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 3. Header UI with specific classes
st.markdown("<h1>🐾 AI Pet Detector</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Project Web Deploy • Detect Cat & Dog</div>", unsafe_allow_html=True)
st.markdown("<p class='author-text'>By Nguyễn Đông Phương - 2286400025</p>", unsafe_allow_html=True)

# 4. Load AI Model with spinner
@st.cache_resource
def load_model():
    # Use MobileNetV2 pre-trained on ImageNet
    model = MobileNetV2(weights='imagenet')
    return model

with st.spinner('✨ Initializing AI Pet Model... this may take a moment.'):
    model = load_model()

# 5. Modern File Upload Section
uploaded_file = st.file_uploader("Upload or drag & drop a clear photo of your pet", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Get image data and convert to standard RGB for predictability
    image_data = Image.open(uploaded_file).convert('RGB')
    
    # Stylized Image Preview within the centered card
    st.image(image_data, use_container_width=True)
    
    # Modernized Prediction Button with hover effect
    if st.button('✨ Phân Tích Hình Ảnh'):
        # Smooth simulated loading animation
        progress_text = "<p class='loading-text'>Scanning visual features...</p>"
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(0.2)
        
        # 4. Image Preprocessing for MobileNetV2
        # Resize to 224x224
        img = image_data.resize((224, 224))
        # Convert to array
        x = image.img_to_array(img)
        # Add batch dimension (1, 224, 224, 3)
        x = np.expand_dims(x, axis=0)
        # Preprocess (normalize pixels)
        x = preprocess_input(x)

        # 5. Prediction
        preds = model.predict(x)
        # Get Top 3 predictions and probabilities
        decoded_preds = decode_predictions(preds, top=3)[0]
        
        # Define comprehensive keyword lists from original code and more
        cat_keywords = ['cat', 'tabby', 'tiger', 'siamese', 'persian', 'lynx', 'leopard', 'kitten', 'cougar', 'lion', 'panther', 'cheetah', 'jaguar']
        dog_keywords = [
            'dog', 'terrier', 'retriever', 'spaniel', 'shepherd', 'hound', 'boxer', 'bulldog', 'dalmatian', 
            'husky', 'corgi', 'pug', 'pomeranian', 'chihuahua', 'beagle', 'collie', 'poodle', 'rottweiler', 
            'doberman', 'shiba', 'akita', 'malamute', 'samoyed', 'chow', 'dane', 'mastiff', 'bernese', 
            'newfoundland', 'schnauzer', 'pinscher', 'sheepdog', 'pointer', 'vizsla', 'setter', 'maltese', 
            'papillon', 'pekingese', 'spitz', 'whippet', 'basenji', 'borzoi', 'greyhound', 'bloodhound', 'wolf'
        ]

        # Process and clean all top 3 predictions
        processed_preds = []
        for pred_raw in decoded_preds:
            _, label_raw, prob_raw = pred_raw
            # Remove underscores for a clean, title-case display
            label_clean = label_raw.replace('_', ' ').title()
            
            category = "Other"
            if any(k in label_raw.lower() for k in cat_keywords):
                category = "Cat"
            elif any(k in label_raw.lower() for k in dog_keywords):
                category = "Dog"
            
            processed_preds.append((label_clean, category, prob_raw))

        # Get top result for display
        top_label, top_cat, top_prob = processed_preds[0]
        
        # Determine main badge styling
        badge_class = "badge-unknown"
        emoji = "❓"
        if top_cat == "Cat":
            emoji = "🐱"
            badge_class = "badge-cat"
        elif top_cat == "Dog":
            emoji = "🐶"
            badge_class = "badge-dog"

        # Display stylized main result badge with bounce animation
        st.markdown(f'<div class="result-badge {badge_class}">{emoji} {top_cat} - {top_label}</div>', unsafe_allow_html=True)
        
        # Display smooth confidence progress bar
        st.progress(float(top_prob), text=f"Confidence: {top_prob:.2%}")

        # Display stylized list of Top 2 & 3 predictions for better insight (improves 'accuracy' through transparency)
        if len(processed_preds) > 1:
            st.markdown("<p class='prediction-details-title'>Other Predictions for Context:</p>", unsafe_allow_html=True)
            st.markdown("<div class='other-predictions-list'>", unsafe_allow_html=True)
            for label, cat, prob in processed_preds[1:]:
                other_emoji = "🐱" if cat == "Cat" else "🐶" if cat == "Dog" else "❓"
                st.markdown(
                    f'<div class="other-pred-item">'
                    f'    <span class="other-pred-label">{other_emoji} {label}</span>'
                    f'    <span class="other-pred-category">({cat})</span>'
                    f'    <span class="other-pred-prob">{prob:.2%}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
