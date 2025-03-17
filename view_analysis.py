import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Threshold Analysis",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("Facial Recognition Threshold Analysis")
st.markdown("""
This page shows the analysis of different threshold values for the facial recognition model.
The graphs help you understand how to set the optimal threshold to balance between false positives and false negatives.
""")

# Display the threshold analysis graph
st.header("Model Performance vs. Threshold")
if os.path.exists("threshold_analysis.png"):
    threshold_img = Image.open("threshold_analysis.png")
    st.image(threshold_img, caption="Model Performance vs. Threshold", use_column_width=True)
    
    st.markdown("""
    ### Interpretation:
    - **Precision** (blue line): Percentage of correct positive predictions. Higher values mean fewer false positives.
    - **Recall** (green line): Percentage of actual positives correctly identified. Higher values mean fewer false negatives.
    - **F1 Score** (red line): Harmonic mean of precision and recall. Higher values indicate better overall performance.
    - **Vertical line**: The optimal threshold that maximizes the F1 score.
    
    As you increase the threshold:
    - Precision increases (fewer false matches)
    - Recall decreases (more false non-matches)
    """)
else:
    st.error("Threshold analysis graph not found. Please run analyze_threshold.py first.")

# Display the score distribution graph
st.header("Distribution of Similarity Scores")
if os.path.exists("score_distribution.png"):
    dist_img = Image.open("score_distribution.png")
    st.image(dist_img, caption="Distribution of Similarity Scores", use_column_width=True)
    
    st.markdown("""
    ### Interpretation:
    - **Green histogram**: Distribution of similarity scores for matching pairs (same person)
    - **Red histogram**: Distribution of similarity scores for non-matching pairs (different people)
    - **Vertical line**: The optimal threshold that separates the two distributions
    
    Ideally:
    - The green distribution should be mostly to the right of the threshold
    - The red distribution should be mostly to the left of the threshold
    - Minimal overlap between the two distributions
    
    If you're experiencing false matches:
    - Look at where the red distribution extends beyond the threshold
    - Consider increasing the threshold to reduce false positives
    """)
else:
    st.error("Score distribution graph not found. Please run analyze_threshold.py first.")

# Threshold adjustment recommendations
st.header("Threshold Adjustment Recommendations")
st.markdown("""
### How to adjust the threshold:

1. **If you're getting false matches** (different people identified as the same):
   - Increase the threshold (try 0.35-0.45)
   - This makes the matching more strict
   - Reduces false positives but may increase false negatives

2. **If you're getting false non-matches** (same person not recognized):
   - Decrease the threshold (try 0.15-0.25)
   - This makes the matching more lenient
   - Reduces false negatives but may increase false positives

3. **For a balanced approach**:
   - Use a threshold around 0.25-0.35
   - This attempts to balance false positives and false negatives

Remember that no threshold will be perfect - there's always a trade-off between false positives and false negatives.
""")

# Footer
st.markdown("---")
st.markdown("Facial Recognition System | Threshold Analysis") 