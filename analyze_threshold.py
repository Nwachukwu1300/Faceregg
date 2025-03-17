import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras.models import load_model

# Define L1Dist layer (needed for model loading)
class L1Dist(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
    
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding) + 1e-6

def load_test_data(test_dir='data', batch_size=16):
    """
    Load test data from directories
    
    Args:
        test_dir: Directory containing test data
        batch_size: Batch size for the dataset
    
    Returns:
        A TensorFlow dataset of test data
    """
    # Define paths
    POS_PATH = os.path.join(test_dir, 'positive')
    NEG_PATH = os.path.join(test_dir, 'negative')
    ANC_PATH = os.path.join(test_dir, 'anchor')
    
    # Check if directories exist
    if not os.path.exists(POS_PATH) or not os.path.exists(NEG_PATH) or not os.path.exists(ANC_PATH):
        print(f"Error: Test data directories not found in {test_dir}")
        print("Please make sure you have 'positive', 'negative', and 'anchor' directories with test images.")
        return None
    
    # Get image file paths
    anchor_images = tf.data.Dataset.list_files(os.path.join(ANC_PATH, '*.jpg'), shuffle=False)
    positive_images = tf.data.Dataset.list_files(os.path.join(POS_PATH, '*.jpg'), shuffle=False)
    negative_images = tf.data.Dataset.list_files(os.path.join(NEG_PATH, '*.jpg'), shuffle=False)
    
    # Count images
    n_anchor = len(list(anchor_images))
    n_positive = len(list(positive_images))
    n_negative = len(list(negative_images))
    
    print(f"Found {n_anchor} anchor images, {n_positive} positive images, and {n_negative} negative images")
    
    # Reset datasets after counting
    anchor_images = tf.data.Dataset.list_files(os.path.join(ANC_PATH, '*.jpg'), shuffle=False)
    positive_images = tf.data.Dataset.list_files(os.path.join(POS_PATH, '*.jpg'), shuffle=False)
    negative_images = tf.data.Dataset.list_files(os.path.join(NEG_PATH, '*.jpg'), shuffle=False)
    
    # Function to preprocess images
    def preprocess_image(file_path):
        # Read image
        img = tf.io.read_file(file_path)
        # Decode image
        img = tf.image.decode_jpeg(img, channels=3)
        # Resize image
        img = tf.image.resize(img, (100, 100))
        # Normalize pixel values
        img = img / 255.0
        return img
    
    # Create positive pairs (anchor-positive)
    positives = tf.data.Dataset.zip((anchor_images, positive_images, tf.data.Dataset.from_tensor_slices(tf.ones(n_positive))))
    
    # Create negative pairs (anchor-negative)
    negatives = tf.data.Dataset.zip((anchor_images, negative_images, tf.data.Dataset.from_tensor_slices(tf.zeros(n_negative))))
    
    # Combine positive and negative pairs
    test_data = positives.concatenate(negatives)
    
    # Map preprocessing function to file paths
    test_data = test_data.map(lambda a, b, y: (preprocess_image(a), preprocess_image(b), y))
    
    # Batch the data
    test_data = test_data.batch(batch_size)
    test_data = test_data.prefetch(8)
    
    return test_data

def analyze_thresholds(model, test_data):
    """
    Analyze different thresholds to find the optimal one
    
    Args:
        model: The trained Siamese model
        test_data: The test dataset
    
    Returns:
        The optimal threshold value
    """
    print("Analyzing thresholds to find optimal value...")
    
    # Collect all predictions
    all_preds = []
    all_labels = []
    
    for batch in test_data:
        X = batch[:2]
        y = batch[2].numpy()
        y_pred = model(X, training=False).numpy().flatten()
        
        all_preds.extend(y_pred)
        all_labels.extend(y)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics at different thresholds
    thresholds = np.arange(0, 1.01, 0.05)
    precisions = []
    recalls = []
    f1_scores = []
    
    print("\nThreshold Analysis Results:")
    print("--------------------------")
    
    for threshold in thresholds:
        binary_preds = (all_preds > threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((binary_preds == 1) & (all_labels == 1))
        fp = np.sum((binary_preds == 1) & (all_labels == 0))
        fn = np.sum((binary_preds == 0) & (all_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        print(f"Threshold {threshold:.2f}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    # Find optimal threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"\nBest threshold: {best_threshold:.4f} (F1 Score: {f1_scores[best_idx]:.4f})")
    
    # Plot metrics vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, 'b-', label='Precision')
    plt.plot(thresholds, recalls, 'g-', label='Recall')
    plt.plot(thresholds, f1_scores, 'r-', label='F1 Score')
    
    # Mark the optimal threshold
    plt.axvline(x=best_threshold, color='k', linestyle='--', 
                label=f'Best Threshold = {best_threshold:.2f}')
    
    plt.title('Model Performance vs. Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('threshold_analysis.png')
    print("Saved threshold analysis plot to 'threshold_analysis.png'")
    
    # Plot prediction distribution
    plt.figure(figsize=(10, 6))
    
    # Separate predictions for positive and negative pairs
    pos_preds = all_preds[all_labels == 1]
    neg_preds = all_preds[all_labels == 0]
    
    # Plot histograms
    plt.hist(pos_preds, bins=20, alpha=0.6, color='green', label='Matching Pairs')
    plt.hist(neg_preds, bins=20, alpha=0.6, color='red', label='Non-matching Pairs')
    
    plt.axvline(x=best_threshold, color='k', linestyle='--', 
                label=f'Best Threshold = {best_threshold:.2f}')
    
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Model Prediction (Similarity Score)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('score_distribution.png')
    print("Saved score distribution plot to 'score_distribution.png'")
    
    return best_threshold

def update_model_metadata(best_threshold):
    """
    Update the model metadata with the best threshold
    
    Args:
        best_threshold: The optimal threshold value
    """
    metadata_path = os.path.join('model', 'metadata.json')
    
    if os.path.exists(metadata_path):
        import json
        
        # Read existing metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update threshold
        metadata['best_threshold'] = float(best_threshold)
        
        # Write updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Updated model metadata with best threshold: {best_threshold:.4f}")
    else:
        print(f"Warning: Metadata file not found at {metadata_path}")

def main():
    # Check if model exists
    model_dir = 'model'
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' not found.")
        print("Please export your model first using export_model.py")
        return
    
    # Load the model
    print(f"Loading model from {model_dir}...")
    try:
        model = load_model(model_dir, custom_objects={'L1Dist': L1Dist})
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Load test data
    test_data = load_test_data()
    if test_data is None:
        return
    
    # Analyze thresholds
    best_threshold = analyze_thresholds(model, test_data)
    
    # Update model metadata
    update_model_metadata(best_threshold)
    
    print("\nThreshold analysis complete!")
    print(f"Optimal threshold: {best_threshold:.4f}")
    print("You can now use this threshold in your app.py or export your model again with:")
    print(f"python export_model.py --checkpoint models/siamese_model_20250310_231530/model_weights.h5 --output model --threshold {best_threshold:.4f}")

if __name__ == "__main__":
    main() 