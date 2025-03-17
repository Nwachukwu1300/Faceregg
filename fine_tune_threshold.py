import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os
from PIL import Image
import io
import matplotlib.pyplot as plt

# Define L1Dist layer (needed for model loading)
class L1Dist(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
    
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding) + 1e-6

# Set page configuration
st.set_page_config(
    page_title="Threshold Fine-Tuning",
    page_icon="🔍",
    layout="wide"
)

# App title and description
st.title("Facial Recognition Threshold Fine-Tuning")
st.markdown("""
This tool helps you fine-tune the threshold for your specific use case by testing different thresholds on image pairs.
Upload two images and see how different thresholds affect the matching decision.
""")

# Function to preprocess images
@st.cache_data
def preprocess_image(img):
    # Convert to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Resize to model input size
    img = cv2.resize(img, (100, 100))
    # Normalize pixel values
    img = img / 255.0
    return img

# Function to load model
@st.cache_resource
def load_model():
    try:
        # Check if model exists in the current directory
        if os.path.exists('model'):
            model = tf.keras.models.load_model(
                'model',
                custom_objects={'L1Dist': L1Dist}
            )
            return model
        else:
            st.error("Model not found. Please make sure the model is in the 'model' directory.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model
model = load_model()

# Create two columns for image upload
col1, col2 = st.columns(2)

with col1:
    st.header("Image 1")
    uploaded_file1 = st.file_uploader("Choose first image", type=["jpg", "jpeg", "png"], key="file1")
    
    if uploaded_file1 is not None:
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(uploaded_file1.read()), dtype=np.uint8)
        image1 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Display the image
        st.image(image1, caption="Uploaded Image 1", use_column_width=True)

with col2:
    st.header("Image 2")
    uploaded_file2 = st.file_uploader("Choose second image", type=["jpg", "jpeg", "png"], key="file2")
    
    if uploaded_file2 is not None:
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(uploaded_file2.read()), dtype=np.uint8)
        image2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Display the image
        st.image(image2, caption="Uploaded Image 2", use_column_width=True)

# Threshold selection
st.header("Threshold Selection")
threshold_options = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
selected_thresholds = st.multiselect(
    "Select thresholds to test",
    threshold_options,
    default=[0.25, 0.35, 0.45]
)

# Analyze button
if st.button("Analyze Images"):
    if uploaded_file1 is None or uploaded_file2 is None:
        st.warning("Please upload both images first.")
    elif model is None:
        st.warning("Model not loaded. Please check if the model exists in the 'model' directory.")
    elif not selected_thresholds:
        st.warning("Please select at least one threshold to test.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Preprocess images
                img1 = preprocess_image(image1)
                img2 = preprocess_image(image2)
                
                # Add batch dimension
                img1_batch = np.expand_dims(img1, axis=0)
                img2_batch = np.expand_dims(img2, axis=0)
                
                # Get prediction
                result = model.predict([img1_batch, img2_batch])
                similarity_score = float(result[0][0])
                
                # Display similarity score
                st.subheader(f"Similarity Score: {similarity_score:.4f}")
                
                # Create a table for threshold analysis
                results = []
                for threshold in selected_thresholds:
                    is_match = similarity_score > threshold
                    results.append({
                        "Threshold": threshold,
                        "Decision": "MATCH" if is_match else "NO MATCH",
                        "Confidence": f"{abs(similarity_score - threshold) / threshold * 100:.1f}%"
                    })
                
                # Display results as a table
                st.table(results)
                
                # Create a visual representation
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot the similarity score as a vertical line
                ax.axvline(x=similarity_score, color='blue', linestyle='-', linewidth=2, 
                          label=f'Similarity Score: {similarity_score:.4f}')
                
                # Plot the thresholds as vertical lines
                for threshold in selected_thresholds:
                    color = 'green' if similarity_score > threshold else 'red'
                    ax.axvline(x=threshold, color=color, linestyle='--', linewidth=1.5,
                              label=f'Threshold: {threshold} ({color})')
                
                # Set the plot limits and labels
                ax.set_xlim(0, 1)
                ax.set_xlabel('Threshold / Similarity Score')
                ax.set_ylabel('Value')
                ax.set_title('Similarity Score vs. Thresholds')
                ax.legend()
                ax.grid(True)
                
                # Display the plot
                st.pyplot(fig)
                
                # Recommendation
                st.subheader("Recommendation")
                if similarity_score < 0.15:
                    st.error("These images are very different. They are definitely not the same person.")
                elif similarity_score < 0.25:
                    st.warning("These images are quite different. They are likely not the same person.")
                elif similarity_score < 0.35:
                    st.info("These images have moderate similarity. They might be the same person in different conditions.")
                elif similarity_score < 0.45:
                    st.success("These images are quite similar. They are likely the same person.")
                else:
                    st.success("These images are very similar. They are almost certainly the same person.")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

# Explanation of thresholds
with st.expander("Understanding Thresholds"):
    st.markdown("""
    ### How to interpret thresholds:
    
    - **Low threshold (0.15-0.25)**: More lenient matching. Higher chance of false positives (different people matched as same).
    - **Medium threshold (0.25-0.35)**: Balanced approach. Moderate false positives and false negatives.
    - **High threshold (0.35-0.45)**: Stricter matching. Higher chance of false negatives (same person not matched).
    - **Very high threshold (0.45+)**: Very strict matching. Only very similar images will match.
    
    ### Tips for choosing the right threshold:
    
    1. **For security applications** (e.g., access control):
       - Use higher thresholds (0.35-0.45) to minimize false positives
       - Better to occasionally reject the right person than admit the wrong person
    
    2. **For convenience applications** (e.g., photo organization):
       - Use lower thresholds (0.15-0.25) to minimize false negatives
       - Better to occasionally group different people than miss grouping the same person
    
    3. **For balanced applications**:
       - Use medium thresholds (0.25-0.35)
       - Try to balance false positives and false negatives
    """)

# Footer
st.markdown("---")
st.markdown("Facial Recognition System | Threshold Fine-Tuning Tool") 