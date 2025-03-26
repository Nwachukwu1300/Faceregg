import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os
from PIL import Image
import io

#define L1Dist layer to load the model
class L1Dist(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
    
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding) + 1e-6

st.set_page_config(
    page_title="Facial Recognition System",
    page_icon="👤",
    layout="wide"
)

st.title("Facial Recognition System")
st.markdown("""
This app uses a Siamese Neural Network to verify if two face images belong to the same person.
Upload two images to see if they match!
""")

with st.sidebar:
    
    st.header("Threshold")
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.35, 0.01)
    st.markdown(f"Images with similarity score > {threshold} will be considered a match.")
    st.markdown("""
    **Note:** Higher threshold values make matching more strict (fewer false positives).
    - 0.15-0.25: Default range (balanced)
    - 0.25-0.35: Moderate strictness
    - 0.35-0.45: High strictness (fewer false matches)
    """)

#function to preprocess images
@st.cache_data
def preprocess_image(img):
    #convert to rgb
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img = cv2.resize(img, (100, 100))
    #normalize pixel values
    img = img / 255.0
    return img

@st.cache_resource
def load_model():
    try:
        if os.path.exists('model'):
            model = tf.keras.models.load_model(
                'model',
                custom_objects={'L1Dist': L1Dist}
            )
            return model
        else:
            st.error("Model not found. Please upload a model first.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model
model = load_model()

#create two columns for image upload
col1, col2 = st.columns(2)

with col1:
    st.header("Image 1")
    uploaded_file1 = st.file_uploader("Choose first image", type=["jpg", "jpeg", "png"], key="file1")
    
    if uploaded_file1 is not None:
        #convert the file to an opencv image
        file_bytes = np.asarray(bytearray(uploaded_file1.read()), dtype=np.uint8)
        image1 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  #convert bgr to rgb
        
        #display the image
        st.image(image1, caption="Uploaded Image 1", use_column_width=True)

with col2:
    st.header("Image 2")
    uploaded_file2 = st.file_uploader("Choose second image", type=["jpg", "jpeg", "png"], key="file2")
    
    if uploaded_file2 is not None:
        file_bytes = np.asarray(bytearray(uploaded_file2.read()), dtype=np.uint8)
        image2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB) 
        
        st.image(image2, caption="Uploaded Image 2", use_column_width=True)

#verify button
if st.button("Verify Faces"):
    if uploaded_file1 is None or uploaded_file2 is None:
        st.warning("Please upload both images first.")
    elif model is None:
        st.warning("Model not loaded. Please check if the model exists in the 'model' directory.")
    else:
        with st.spinner("Verifying..."):
            try:
                #preprocess images
                img1 = preprocess_image(image1)
                img2 = preprocess_image(image2)
                
                #add batch dimension
                img1 = np.expand_dims(img1, axis=0)
                img2 = np.expand_dims(img2, axis=0)
                
                #get prediction
                result = model.predict([img1, img2])
                similarity_score = float(result[0][0])
                
                #determine if match based on threshold
                is_match = similarity_score > threshold
                
                #display result
                st.subheader("Result")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Similarity Score", f"{similarity_score:.4f}")
                
                with col2:
                    if is_match:
                        st.success("✅ MATCH: Same Person")
                    else:
                        st.error("❌ NO MATCH: Different People")
                
                #explanation
                st.markdown(f"""
                **Interpretation:**
                - Score close to 1.0 indicates high similarity (same person)
                - Score close to 0.0 indicates low similarity (different people)
                - Current threshold: {threshold}
                """)
                
            except Exception as e:
                st.error(f"Error during verification: {str(e)}")

#add advanced tools section
st.markdown("---")
st.header("Advanced Tools")
st.markdown("Use these tools to analyze and fine-tune the facial recognition system:")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Threshold Analysis", use_container_width=True):
        #using javascript to open the view_analysis.py in a new tab
        js = f"""
        <script>
        window.open("http://localhost:8503", "_blank");
        </script>
        """
        st.components.v1.html(js, height=0)
        st.info("Opening Threshold Analysis in a new tab... If it doesn't open automatically, please run 'streamlit run view_analysis.py' in your terminal.")

with col2:
    if st.button("🔍 Fine-Tune Threshold", use_container_width=True):
        #using javascript to open the fine_tune_threshold.py in a new tab
        js = f"""
        <script>
        window.open("http://localhost:8504", "_blank");
        </script>
        """
        st.components.v1.html(js, height=0)
        st.info("Opening Threshold Fine-Tuning in a new tab... If it doesn't open automatically, please run 'streamlit run fine_tune_threshold.py' in your terminal.")

#add information about model architecture
with st.expander("Model Architecture Details"):
    st.markdown("""
    ### Siamese Neural Network Architecture
    
    The model uses a Siamese neural network with the following components:
    
    #### Embedding Network:
    - **Input**: 100×100×3 RGB images
    - **First Block**: Conv2D(64, 10×10) → BatchNorm → MaxPool → Dropout(0.3)
    - **Second Block**: Conv2D(128, 3×3) → MaxPool → Dropout(0.3)
    - **Third Block**: Conv2D(128, 7×7) → MaxPool → Dropout(0.3)
    - **Fourth Block**: Conv2D(256, 4×4) → Flatten → Dense(4096)
    - **Output**: 4096-dimensional feature vector
    
    #### Distance Layer:
    - L1 (Manhattan) distance between embedding vectors
    
    #### Classification:
    - Single neuron with sigmoid activation
    - Output: Similarity score between 0 and 1
    """)

#footer
st.markdown("---")
st.markdown("Facial Recognition System using Siamese Neural Networks | Built with TensorFlow and Streamlit") 