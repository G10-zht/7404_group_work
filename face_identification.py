import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from time import time
import cv2

def load_olivetti_dataset():
    """
    Load the Olivetti faces dataset from the archive folder
    """
    print("Loading face images from archive folder...")
    import os
    import glob
    import re
    
    # Get all image files in the archive folder
    image_files = glob.glob(os.path.join('archive', '*.jpg'))
    
    if not image_files:
        raise FileNotFoundError("No image files found in the archive folder")
    
    # Sort image files to ensure consistent ordering
    image_files.sort()
    
    X = []  # Flattened image data
    y = []  # Person IDs (labels)
    X_images = []  # Images for visualization
    
    # Target size for images (Olivetti faces are 64x64)
    target_size = (64, 64)
    
    # Process each image file
    for image_path in image_files:
        # Extract person ID from filename (format: {image_number}_{person_id}.jpg)
        filename = os.path.basename(image_path)
        match = re.search(r'_(\d+)\.jpg$', filename)
        if match:
            person_id = int(match.group(1)) - 1  # Subtract 1 to make IDs 0-based
            
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Failed to load image: {image_path}")
                continue
            
            # Resize to match Olivetti faces size
            img_resized = cv2.resize(img, target_size)
            
            # Apply preprocessing steps
            img_denoised = cv2.GaussianBlur(img_resized, (3, 3), 0)
            img_enhanced = cv2.equalizeHist(img_denoised)
            img_normalized = img_enhanced / 255.0
            
            # Add to datasets
            X.append(img_normalized.flatten())
            y.append(person_id)
            X_images.append(img_normalized)
    
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    X_images = np.array(X_images)
    
    print(f"Loaded {len(X)} images of {len(np.unique(y))} different people")
    
    return X, y, X_images

def preprocess_images(X, X_images):
    """
    Preprocess the images: normalize and enhance
    """
    print("Preprocessing images...")
    X_preprocessed = []
    
    for i, img in enumerate(X_images):
        # Convert to uint8 for OpenCV operations
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Apply noise reduction (Gaussian blur)
        img_denoised = cv2.GaussianBlur(img_uint8, (3, 3), 0)
        
        # Apply histogram equalization for enhancement
        img_enhanced = cv2.equalizeHist(img_denoised)
        
        # Normalize back to [0, 1]
        img_normalized = img_enhanced / 255.0
        
        X_preprocessed.append(img_normalized.flatten())
    
    return np.array(X_preprocessed)

def extract_hog_features(X_images):
    """
    Extract HOG features from images
    """
    print("Extracting HOG features...")
    hog_features = []
    
    for img in X_images:
        # Convert to uint8 for HOG
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Extract HOG features
        # Parameters based on the paper: 8x8 pixels per cell, 2x2 cells per block
        features, hog_image = hog(
            img_uint8, 
            orientations=9, 
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), 
            block_norm='L2-Hys',
            visualize=True
        )
        
        hog_features.append(features)
    
    return np.array(hog_features)

def apply_pca(X, n_components=150):
    """
    Apply PCA for dimensionality reduction
    """
    print(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Calculate variance explained
    variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)
    
    print(f"Variance explained by {n_components} components: {cumulative_variance[-1]:.4f}")
    
    return X_pca, pca

def train_svm_with_grid_search(X_train, y_train, kernel_options=['linear', 'rbf']):
    """
    Train SVM classifier with grid search for hyperparameter optimization
    """
    print("Training SVM with grid search...")
    param_grid = {
        'C': [1, 10, 100, 500, 1000],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
        'kernel': kernel_options
    }
    
    grid_search = GridSearchCV(
        SVC(probability=True), 
        param_grid, 
        cv=5, 
        n_jobs=-1, 
        verbose=1
    )
    
    start_time = time()
    grid_search.fit(X_train, y_train)
    training_time = time() - start_time
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Training time: {training_time:.2f} seconds")
    
    return grid_search.best_estimator_, training_time

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data
    """
    print("Evaluating model...")
    start_time = time()
    y_pred = model.predict(X_test)
    prediction_time = time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Prediction time: {prediction_time:.4f} seconds")
    
    return accuracy, f1, prediction_time, y_pred

def visualize_results(X_images_test, y_test, y_pred, indices_to_show=5):
    """
    Visualize some test results
    """
    plt.figure(figsize=(15, 5))
    
    for i in range(indices_to_show):
        plt.subplot(1, indices_to_show, i+1)
        plt.imshow(X_images_test[i], cmap='gray')
        plt.title(f"True: {y_test[i]}\nPred: {y_pred[i]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('face_identification_results.png')
    plt.close()

def compare_methods(hog_results, pca_results):
    """
    Compare HOG and PCA methods
    """
    methods = ['HOG', 'PCA']
    metrics = {
        'Feature Extraction Time (s)': [hog_results['feature_time'], pca_results['feature_time']],
        'Training Time (s)': [hog_results['training_time'], pca_results['training_time']],
        'Prediction Time (s)': [hog_results['prediction_time'], pca_results['prediction_time']],
        'Accuracy': [hog_results['accuracy'], pca_results['accuracy']],
        'F1 Score': [hog_results['f1'], pca_results['f1']]
    }
    
    # Print comparison table
    print("\nComparison of HOG and PCA methods:")
    print("-" * 60)
    print(f"{'Metric':<25} {'HOG':<15} {'PCA':<15}")
    print("-" * 60)
    
    for metric, values in metrics.items():
        if 'Time' in metric:
            print(f"{metric:<25} {values[0]:<15.4f} {values[1]:<15.4f}")
        else:
            print(f"{metric:<25} {values[0]:<15.4f} {values[1]:<15.4f}")
    
    print("-" * 60)
    
    # Create bar chart for comparison
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy and F1 score
    plt.subplot(2, 1, 1)
    performance_metrics = ['Accuracy', 'F1 Score']
    performance_values = [
        [metrics['Accuracy'][0], metrics['F1 Score'][0]],  # HOG
        [metrics['Accuracy'][1], metrics['F1 Score'][1]]   # PCA
    ]
    
    x = np.arange(len(performance_metrics))
    width = 0.35
    
    plt.bar(x - width/2, performance_values[0], width, label='HOG')
    plt.bar(x + width/2, performance_values[1], width, label='PCA')
    plt.ylabel('Score')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x, performance_metrics)
    plt.ylim(0, 1.1)
    plt.legend()
    
    # Plot time metrics
    plt.subplot(2, 1, 2)
    time_metrics = ['Feature Extraction Time', 'Training Time', 'Prediction Time']
    time_values = [
        [metrics['Feature Extraction Time (s)'][0], metrics['Training Time (s)'][0], metrics['Prediction Time (s)'][0]],  # HOG
        [metrics['Feature Extraction Time (s)'][1], metrics['Training Time (s)'][1], metrics['Prediction Time (s)'][1]]   # PCA
    ]
    
    x = np.arange(len(time_metrics))
    
    plt.bar(x - width/2, time_values[0], width, label='HOG')
    plt.bar(x + width/2, time_values[1], width, label='PCA')
    plt.ylabel('Time (seconds)')
    plt.title('Time Metrics Comparison')
    plt.xticks(x, time_metrics)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hog_pca_comparison.png')
    plt.close()

def main():
    """
    Main function to run the face identification experiment
    """
    # Step 1: Load the Olivetti dataset
    X, y, X_images = load_olivetti_dataset()
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Step 2: Preprocess the images
    X_preprocessed = preprocess_images(X, X_images)
    
    # Step 3: Split the data into training and testing sets
    # For each person, use 9 images for training and 1 for testing (as per paper)
    X_train, X_test, y_train, y_test, X_images_train, X_images_test = [], [], [], [], [], []
    
    for person_id in range(40):  # 40 different people in the dataset
        person_indices = np.where(y == person_id)[0]
        np.random.shuffle(person_indices)
        
        # Select 9 images for training and 1 for testing
        train_indices = person_indices[:9]
        test_indices = person_indices[9:]
        
        X_train.extend(X_preprocessed[train_indices])
        X_test.extend(X_preprocessed[test_indices])
        y_train.extend(y[train_indices])
        y_test.extend(y[test_indices])
        X_images_train.extend(X_images[train_indices])
        X_images_test.extend(X_images[test_indices])
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    X_images_train = np.array(X_images_train)
    X_images_test = np.array(X_images_test)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Step 4a: Extract HOG features
    start_time = time()
    X_train_hog = extract_hog_features(X_images_train)
    X_test_hog = extract_hog_features(X_images_test)
    hog_feature_time = time() - start_time
    print(f"HOG features extracted: {X_train_hog.shape[1]} features per image")
    print(f"HOG feature extraction time: {hog_feature_time:.2f} seconds")
    
    # Step 4b: Apply PCA
    start_time = time()
    X_train_pca, pca_model = apply_pca(X_train, n_components=150)
    X_test_pca = pca_model.transform(X_test)
    pca_feature_time = time() - start_time
    print(f"PCA features extracted: {X_train_pca.shape[1]} features per image")
    print(f"PCA feature extraction time: {pca_feature_time:.2f} seconds")
    
    # Step 5a: Train SVM on HOG features
    print("\n--- Training SVM on HOG features ---")
    svm_hog, hog_training_time = train_svm_with_grid_search(X_train_hog, y_train)
    
    # Step 5b: Train SVM on PCA features
    print("\n--- Training SVM on PCA features ---")
    svm_pca, pca_training_time = train_svm_with_grid_search(X_train_pca, y_train)
    
    # Step 6a: Evaluate HOG-SVM model
    print("\n--- Evaluating HOG-SVM model ---")
    hog_accuracy, hog_f1, hog_prediction_time, hog_predictions = evaluate_model(svm_hog, X_test_hog, y_test)
    
    # Step 6b: Evaluate PCA-SVM model
    print("\n--- Evaluating PCA-SVM model ---")
    pca_accuracy, pca_f1, pca_prediction_time, pca_predictions = evaluate_model(svm_pca, X_test_pca, y_test)
    
    # Step 7: Visualize some results
    print("\n--- Visualizing results ---")
    visualize_results(X_images_test, y_test, hog_predictions, indices_to_show=5)
    
    # Step 8: Compare methods
    hog_results = {
        'feature_time': hog_feature_time,
        'training_time': hog_training_time,
        'prediction_time': hog_prediction_time,
        'accuracy': hog_accuracy,
        'f1': hog_f1
    }
    
    pca_results = {
        'feature_time': pca_feature_time,
        'training_time': pca_training_time,
        'prediction_time': pca_prediction_time,
        'accuracy': pca_accuracy,
        'f1': pca_f1
    }
    
    compare_methods(hog_results, pca_results)
    
    print("\nExperiment completed successfully!")
    print("Results saved as 'face_identification_results.png' and 'hog_pca_comparison.png'")

if __name__ == "__main__":
    # main()
    X, y, X_images = load_olivetti_dataset()
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
