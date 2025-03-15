# Siamese Neural Network Architecture for Facial Recognition

This document provides a detailed explanation of the Siamese neural network architecture implemented in the facial recognition system.

## Overview

The system uses a Siamese neural network architecture to perform facial verification (determining if two face images belong to the same person). Siamese networks are particularly well-suited for this task because they:

1. Learn to extract meaningful features from faces
2. Compare these features to determine similarity
3. Can generalize well even with limited training data per identity

## Architecture Components

The complete architecture consists of three main components:

1. **Embedding Network**: Extracts feature vectors from face images
2. **Distance Layer**: Computes the distance between feature vectors
3. **Classification Layer**: Determines if the faces match based on the distance

### 1. Embedding Network

The embedding network processes each input image and extracts a feature vector (embedding):

```
def make_embedding():
    inp=Input(shape=(100,100,3), name='input_image')

    #First block 
    c1 = Conv2D(64, (10,10), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(inp)
    c1 = BatchNormalization()(c1)
    m1 = MaxPooling2D(64,(2,2),padding='same')(c1)
    m1 = tf.keras.layers.Dropout(0.3)(m1)

    #Second block
    c2 = Conv2D(128, (3,3), activation='relu')(m1)
    m2 = MaxPooling2D(64,(2,2),padding='same')(c2)
    m2 = tf.keras.layers.Dropout(0.3)(m2)

    #Third block 
    c3 = Conv2D(128, (7,7), activation='relu')(m2)
    m3 = MaxPooling2D(64,(2,2),padding='same')(c3)
    m3 = tf.keras.layers.Dropout(0.3)(m3)

    #Fourth block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')
```

#### Architecture Details:

- **Input**: 100×100×3 RGB images

- **First Convolutional Block**:
  - Conv2D: 64 filters, 10×10 kernel, ReLU activation
  - L2 regularization (0.01) to prevent overfitting
  - BatchNormalization for training stability
  - MaxPooling2D: 2×2 pool size with same padding
  - Dropout: 30% to prevent overfitting

- **Second Convolutional Block**:
  - Conv2D: 128 filters, 3×3 kernel, ReLU activation
  - MaxPooling2D: 2×2 pool size with same padding
  - Dropout: 30%

- **Third Convolutional Block**:
  - Conv2D: 128 filters, 7×7 kernel, ReLU activation
  - MaxPooling2D: 2×2 pool size with same padding
  - Dropout: 30%

- **Fourth Convolutional Block**:
  - Conv2D: 256 filters, 4×4 kernel, ReLU activation
  - Flatten: Converts the 3D feature maps to a 1D vector
  - Dense: 4096 units with sigmoid activation

- **Output**: 4096-dimensional feature vector (embedding)

#### Design Considerations:

- **Variable Kernel Sizes**: The network uses different kernel sizes (10×10, 3×3, 7×7, 4×4) to capture features at different scales.
- **Increasing Filter Counts**: The number of filters increases from 64 to 256 as we go deeper, allowing the network to learn more complex features.
- **Regularization**: L2 regularization, batch normalization, and dropout are used to prevent overfitting.
- **Sigmoid Activation**: The final layer uses sigmoid activation to constrain the embedding values between 0 and 1.

### 2. Distance Layer

The distance layer computes the L1 (Manhattan) distance between the embedding vectors of two images:

```python
class L1Dist(Layer):
    #Init method for inheritance
    def __init__(self, **kwargs):
        super().__init__()

    #Des the similarity calculation 
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding) + 1e-6
```

#### Key Features:

- **L1 Distance**: Calculates the absolute difference between corresponding elements of the embedding vectors.
- **Numerical Stability**: Adds a small epsilon (1e-6) to prevent numerical issues.
- **Custom TensorFlow Layer**: Implemented as a custom layer that can be integrated into the model.

### 3. Complete Siamese Network

The complete Siamese network combines the embedding network and distance layer:

```python
def make_siamese_model():
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))

    #Validation image in the network
    validation_image = Input(name='validation_img', shape=(100,100,3))

    #Combine siamese distance components
    siamise_layer = L1Dist()
    siamise_layer._name = 'distance'
    distances = siamise_layer(embedding(input_image), embedding(validation_image))

    #Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
```

#### Architecture Flow:

1. **Two Input Branches**:
   - Both the anchor image and validation image are processed through the same embedding network (shared weights).
   - This weight sharing is a key characteristic of Siamese networks.

2. **Distance Calculation**:
   - The L1 distance between the two embeddings is calculated.
   - This distance represents how different the two face embeddings are.

3. **Classification**:
   - A single dense neuron with sigmoid activation produces a similarity score between 0 and 1.
   - Values close to 1 indicate the same person (match).
   - Values close to 0 indicate different people (no match).

## Model Parameters and Complexity

- **Total Parameters**: Approximately 4.4 million parameters
- **Input Shape**: Two 100×100×3 RGB images
- **Output Shape**: Single scalar value between 0 and 1
- **Embedding Dimension**: 4096

## Training Configuration

- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam with learning rate scheduling
  - Initial learning rate: 5e-5
  - Exponential decay: 0.97 every 200 steps
- **Gradient Clipping**: Gradients clipped to [-1.0, 1.0] to prevent exploding gradients
- **Batch Size**: 32
- **Epochs**: 50 (recommended)

## Decision Threshold

The model outputs a similarity score between 0 and 1. To make a binary decision (match/no match), a threshold is applied:

- **Default Threshold**: 0.5
- **Optimal Threshold**: Determined through analysis to maximize F1 score (typically around 0.3-0.4)

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

## References

- The architecture is inspired by the original Siamese Network paper: [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- Modern improvements include batch normalization, dropout, and learning rate scheduling. 