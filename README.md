# Facial Recognition System

A facial verification system using Siamese Neural Networks implemented in TensorFlow. This system determines whether two face images belong to the same person.

## Getting Started

### Download the Model
```bash
# Download pre-trained model from HuggingFace
https://huggingface.co/mmesomaa/facereg/tree/main/model
```

### Install Requirements
```bash
pip install -r requirements.txt
# Required: tensorflow>=2.4.0, opencv-python>=4.5.1, numpy>=1.19.5, streamlit>=1.8.0
```

### Run the Streamlit App
```bash
streamlit run app.py
```
The app will launch at `http://localhost:8501` where you can upload images.

## Overview

This project implements a complete facial verification pipeline:
1. Data collection from webcam
2. Data preprocessing and augmentation
3. Siamese neural network architecture
4. Model training and evaluation
5. Deployment-ready model saving

## System Components

### Data Collection

The system includes a webcam-based data collection module that:
- Captures images from the webcam in real-time
- Detects faces using Haar Cascade classifier
- Crops and saves images to appropriate directories
- Supports both manual and automatic capture modes

#### Image Categories:
- **Anchor Images**: Reference images of the target person
- **Positive Images**: Different images of the same person
- **Negative Images**: Images of different people (from external sources)

### Data Preprocessing

The preprocessing pipeline includes:
- Image loading and decoding
- Data augmentation (random flips, brightness and contrast adjustments)
- Resizing to 100×100 pixels
- Pixel normalization (0-1 range)
- Creation of balanced positive and negative pairs


## Usage Instructions

### Creating Your Own Dataset
```bash
python collect_data.py
# Press 'a' to capture an anchor image
# Press 'p' to capture a positive image
# Press 'c' to toggle automatic capture mode
# Press 'q' to quit
```

### Model Training
```python
# Train the model
train(train_data, EPOCHS)
```

### Face Verification
```python
def verify_faces(img_path1, img_path2, threshold=best_threshold):
    # Preprocess images
    img1 = preprocess_image(img_path1)
    img2 = preprocess_image(img_path2)
    
    # Add batch dimension
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    
    # Get prediction
    result = model.predict([img1, img2])
    similarity_score = result[0][0]
    
    # Determine if match based on optimal threshold
    is_match = similarity_score > threshold
    
    return {'is_match': bool(is_match), 'score': float(similarity_score)}
```

## Implementation Details

### Key Features
- **Efficient Data Pipeline**: Optimized TensorFlow data pipeline with caching and prefetching
- **Data Augmentation**: Random flips, brightness and contrast adjustments for better generalization
- **Regularization**: L2 regularization and dropout to prevent overfitting
- **Learning Rate Scheduling**: Exponential decay to fine-tune training
- **Threshold Optimization**: Analysis to find the optimal decision threshold

### Performance Considerations
- GPU acceleration with memory growth enabled
- Batch processing for efficient training
- Gradient clipping to prevent exploding gradients


## Visualization

The Siamese network architecture can be visualized as follows:

```
Input Image 1 (100×100×3)    Input Image 2 (100×100×3)
        │                           │
        ▼                           ▼
┌─────────────────┐       ┌─────────────────┐
│                 │       │                 │
│  Embedding      │       │  Embedding      │
│  Network        │       │  Network        │
│  (Shared        │       │  (Shared        │
│   Weights)      │       │   Weights)      │
│                 │       │                 │
└─────────────────┘       └─────────────────┘
        │                           │
        ▼                           ▼
   Embedding 1                 Embedding 2
   (4096-dim)                  (4096-dim)
        │                           │
        └────────────┬─────────────┘
                     ▼
             ┌───────────────┐
             │ L1 Distance   │
             │ Layer         │
             └───────────────┘
                     │
                     ▼
             ┌───────────────┐
             │ Dense Layer   │
             │ (1 unit,      │
             │  sigmoid)     │
             └───────────────┘
                     │
                     ▼
             Similarity Score
                  (0-1)
```

## Troubleshooting

### Common Issues
1. **Model Loading Error**: Ensure correct TensorFlow version and model files
2. **Webcam Access**: Grant browser permission to access your webcam
3. **Low Accuracy**: Ensure good lighting and proper face positioning

## References
- [Siamese Network Paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) - Original paper on Siamese networks for one-shot learning
