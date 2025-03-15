# Facial Recognition System

A facial verification system using Siamese Neural Networks implemented in TensorFlow. This system can determine whether two face images belong to the same person.

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

### Model Architecture

#### Embedding Network
The embedding network processes each input image and extracts feature vectors:

- **Input**: 100×100×3 RGB images
- **Architecture**:
  - **First Block**:
    - Conv2D: 64 filters, 10×10 kernel, ReLU activation with L2 regularization
    - BatchNormalization
    - MaxPooling2D: 2×2 pool size
    - Dropout: 30%
  
  - **Second Block**:
    - Conv2D: 128 filters, 3×3 kernel, ReLU activation
    - MaxPooling2D: 2×2 pool size
    - Dropout: 30%
  
  - **Third Block**:
    - Conv2D: 128 filters, 7×7 kernel, ReLU activation
    - MaxPooling2D: 2×2 pool size
    - Dropout: 30%
  
  - **Fourth Block**:
    - Conv2D: 256 filters, 4×4 kernel, ReLU activation
    - Flatten
    - Dense: 4096 units, sigmoid activation
  
- **Output**: 4096-dimensional feature vector

#### Distance Layer
A custom layer that computes the L1 distance (Manhattan distance) between the feature vectors of two images.

#### Siamese Network
The complete model that:
- Takes two input images (anchor and comparison)
- Processes both through the same embedding network (shared weights)
- Computes the L1 distance between their embeddings
- Passes the distance through a sigmoid activation to output a similarity score (0-1)

### Training Process

- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam with learning rate scheduling (starting at 5e-5)
- **Training Data**: Balanced positive and negative pairs
- **Validation**: 30% of data reserved for validation
- **Checkpointing**: Model weights saved every 10 epochs

### Evaluation Metrics

The model is evaluated using:
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to find all positive matches
- **F1 Score**: Harmonic mean of precision and recall
- **Threshold Analysis**: Finding optimal decision boundary

### Model Saving

The system saves:
- Full model architecture and weights
- Embedding model for feature extraction
- Metadata including optimal threshold
- Usage example code for deployment

## Usage Instructions

### Data Collection
```python
# Run the data collection script
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

## References
- [Siamese Network Paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) - Original paper on Siamese networks for one-shot learning 