import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import os
from face_identification import load_olivetti_dataset, preprocess_images, extract_hog_features, apply_pca, train_svm_with_grid_search

def load_and_preprocess_custom_image(image_path, target_size=(64, 64)):
    """
    Load and preprocess a custom face image
    """
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize to match Olivetti faces size
    img_resized = cv2.resize(img, target_size)
    
    # Apply preprocessing steps
    img_denoised = cv2.GaussianBlur(img_resized, (3, 3), 0)
    img_enhanced = cv2.equalizeHist(img_denoised)
    img_normalized = img_enhanced / 255.0
    
    return img_normalized

def train_and_save_models():
    """
    Train HOG-SVM and PCA-SVM models and save them to disk
    """
    # Load dataset
    X, y, X_images = load_olivetti_dataset()
    
    # Preprocess images
    X_preprocessed = preprocess_images(X, X_images)
    
    # Split data (using all data for training in this case)
    X_train, y_train = X_preprocessed, y
    X_images_train = X_images
    
    # Extract HOG features
    X_train_hog = extract_hog_features(X_images_train)
    
    # Apply PCA
    X_train_pca, pca_model = apply_pca(X_train, n_components=150)
    
    # Train SVM on HOG features
    print("Training HOG-SVM model...")
    svm_hog, _ = train_svm_with_grid_search(X_train_hog, y_train)
    
    # Train SVM on PCA features
    print("Training PCA-SVM model...")
    svm_pca, _ = train_svm_with_grid_search(X_train_pca, y_train)
    
    # Save models
    print("Saving models...")
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save HOG-SVM model
    with open('models/hog_svm_model.pkl', 'wb') as f:
        pickle.dump(svm_hog, f)
    
    # Save PCA model and PCA-SVM model
    with open('models/pca_svm_model.pkl', 'wb') as f:
        pickle.dump((pca_model, svm_pca), f)
    
    print("Models saved successfully in the 'models' directory!")
    
    return svm_hog, pca_model, svm_pca

def predict_with_saved_models(image_path):
    """
    Predict the identity of a face using saved models
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
    
    # Check if models exist, if not train them
    if not (os.path.exists(model_paths['hog']) and os.path.exists(model_paths['pca'])):
        print("Models not found. Training new models...")
        svm_hog, pca_model, svm_pca = train_and_save_models()
    else:
        # Load models
        print("Loading saved models...")
        with open(model_paths['hog'], 'rb') as f:
            svm_hog = pickle.load(f)
        
        with open(model_paths['pca'], 'rb') as f:
            pca_model, svm_pca = pickle.load(f)
    
    # Load and preprocess the custom image
    img = load_and_preprocess_custom_image(image_path)
    
    # Extract HOG features
    img_hog = extract_hog_features(np.array([img]))
    
    # Apply PCA
    img_pca = pca_model.transform(img.flatten().reshape(1, -1))
    
    # Make predictions
    hog_pred = svm_hog.predict(img_hog)[0]
    hog_prob = np.max(svm_hog.predict_proba(img_hog))
    
    pca_pred = svm_pca.predict(img_pca)[0]
    pca_prob = np.max(svm_pca.predict_proba(img_pca))
    
    # Display results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Input Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.text(0.1, 0.7, f"HOG-SVM Prediction: Person {hog_pred}", fontsize=12)
    plt.text(0.1, 0.6, f"Confidence: {hog_prob:.2f}", fontsize=12)
    plt.text(0.1, 0.4, f"PCA-SVM Prediction: Person {pca_pred}", fontsize=12)
    plt.text(0.1, 0.3, f"Confidence: {pca_prob:.2f}", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('custom_prediction_result.png')
    plt.show()
    
    print(f"HOG-SVM Prediction: Person {hog_pred} (Confidence: {hog_prob:.2f})")
    print(f"PCA-SVM Prediction: Person {pca_pred} (Confidence: {pca_prob:.2f})")
    print("Prediction visualization saved as 'custom_prediction_result.png'")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict identity from a face image')
    parser.add_argument('image_path', type=str, help='Path to the face image')
    parser.add_argument('--train', action='store_true', help='Force retraining of models')
    
    args = parser.parse_args()
    
    if args.train:
        train_and_save_models()
    
    predict_with_saved_models(args.image_path)
