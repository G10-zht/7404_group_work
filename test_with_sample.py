import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
import pickle
import os
from face_identification import extract_hog_features, apply_pca, train_svm_with_grid_search,load_olivetti_dataset

def load_sample_image():
    """
    Load a sample image from the Olivetti dataset
    """
    print("Loading a sample image from Olivetti dataset...")
    dataset = fetch_olivetti_faces(shuffle=True, random_state=42)
    
    # Get a random sample image
    sample_idx = np.random.randint(0, dataset.images.shape[0])
    sample_image = dataset.images[sample_idx]
    sample_target = dataset.target[sample_idx]
    
    print(f"Selected sample image of person {sample_target} (index {sample_idx})")
    
    return sample_image, sample_target

def test_with_sample_image():
    """
    Test the face identification system with a sample image
    """
    # Check if models exist in the models directory
    model_paths = {
        'hog': 'models/hog_svm_model.pkl',
        'pca': 'models/pca_svm_model.pkl'
    }
    
    # Check if models exist in the root directory (for backward compatibility)
    if not (os.path.exists(model_paths['hog']) and os.path.exists(model_paths['pca'])):
        model_paths = {
            'hog': 'hog_svm_model.pkl',
            'pca': 'pca_svm_model.pkl'
        }
        
        if not (os.path.exists(model_paths['hog']) and os.path.exists(model_paths['pca'])):
            print("Models not found. Please run face_identification.py or save_trained_models.py first to train the models.")
            return
    
    # Load models
    print("Loading saved models...")
    with open(model_paths['hog'], 'rb') as f:
        svm_hog = pickle.load(f)
    
    with open(model_paths['pca'], 'rb') as f:
        pca_model, svm_pca = pickle.load(f)
    
    # Load a sample image
    sample_image, true_label = load_sample_image()
    
    # Extract HOG features
    sample_hog = extract_hog_features(np.array([sample_image]))
    
    # Apply PCA
    sample_pca = pca_model.transform(sample_image.flatten().reshape(1, -1))
    
    # Make predictions
    hog_pred = svm_hog.predict(sample_hog)[0]
    hog_prob = np.max(svm_hog.predict_proba(sample_hog))
    
    pca_pred = svm_pca.predict(sample_pca)[0]
    pca_prob = np.max(svm_pca.predict_proba(sample_pca))
    
    # Display results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(sample_image, cmap='gray')
    plt.title(f"Sample Image\nTrue Label: Person {true_label}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.text(0.1, 0.8, f"True Label: Person {true_label}", fontsize=12)
    plt.text(0.1, 0.7, f"HOG-SVM Prediction: Person {hog_pred}", fontsize=12)
    plt.text(0.1, 0.6, f"HOG Confidence: {hog_prob:.2f}", fontsize=12)
    plt.text(0.1, 0.4, f"PCA-SVM Prediction: Person {pca_pred}", fontsize=12)
    plt.text(0.1, 0.3, f"PCA Confidence: {pca_prob:.2f}", fontsize=12)
    
    # Add accuracy indicators
    hog_correct = hog_pred == true_label
    pca_correct = pca_pred == true_label
    
    plt.text(0.1, 0.1, f"HOG Prediction: {'✓' if hog_correct else '✗'}", 
             fontsize=14, color='green' if hog_correct else 'red')
    plt.text(0.6, 0.1, f"PCA Prediction: {'✓' if pca_correct else '✗'}", 
             fontsize=14, color='green' if pca_correct else 'red')
    
    plt.tight_layout()
    plt.savefig('sample_test_result.png')
    plt.show()
    
    print(f"True Label: Person {true_label}")
    print(f"HOG-SVM Prediction: Person {hog_pred} (Confidence: {hog_prob:.2f}) - {'Correct' if hog_correct else 'Incorrect'}")
    print(f"PCA-SVM Prediction: Person {pca_pred} (Confidence: {pca_prob:.2f}) - {'Correct' if pca_correct else 'Incorrect'}")
    print("Test visualization saved as 'sample_test_result.png'")

if __name__ == "__main__":
    test_with_sample_image()
