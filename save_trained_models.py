import numpy as np
import pickle
import os
from face_identification import load_olivetti_dataset, preprocess_images, extract_hog_features, apply_pca, train_svm_with_grid_search

def train_and_save_models():
    """
    Train HOG-SVM and PCA-SVM models on the full Olivetti dataset and save them to disk
    """
    print("Loading Olivetti faces dataset...")
    X, y, X_images = load_olivetti_dataset()
    
    print("Preprocessing images...")
    X_preprocessed = preprocess_images(X, X_images)
    
    # Extract HOG features
    print("Extracting HOG features...")
    X_hog = extract_hog_features(X_images)
    
    # Apply PCA
    print("Applying PCA...")
    X_pca, pca_model = apply_pca(X_preprocessed, n_components=150)
    
    # Train SVM on HOG features
    print("\n--- Training SVM on HOG features ---")
    svm_hog, hog_training_time = train_svm_with_grid_search(X_hog, y)
    
    # Train SVM on PCA features
    print("\n--- Training SVM on PCA features ---")
    svm_pca, pca_training_time = train_svm_with_grid_search(X_pca, y)
    
    # Save models
    print("\nSaving models...")
    
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
    print(f"HOG-SVM model saved as 'models/hog_svm_model.pkl'")
    print(f"PCA model and PCA-SVM model saved as 'models/pca_svm_model.pkl'")
    
    # Return models for potential further use
    return svm_hog, pca_model, svm_pca

if __name__ == "__main__":
    train_and_save_models()
