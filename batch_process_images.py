import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import os
import glob
from face_identification import extract_hog_features

def load_models():
    """
    Load the trained models from disk
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
    
    # Check if models exist
    if not (os.path.exists(model_paths['hog']) and os.path.exists(model_paths['pca'])):
        print("Models not found. Please run face_identification.py or save_trained_models.py first to train the models.")
        return None, None, None
    
    # Load models
    print("Loading saved models...")
    with open(model_paths['hog'], 'rb') as f:
        svm_hog = pickle.load(f)
    
    with open(model_paths['pca'], 'rb') as f:
        pca_model, svm_pca = pickle.load(f)
    
    return svm_hog, pca_model, svm_pca

def load_and_preprocess_image(image_path, target_size=(64, 64)):
    """
    Load and preprocess an image
    """
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found: {image_path}")
        return None
    
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Failed to load image: {image_path}")
        return None
    
    # Resize to match Olivetti faces size
    img_resized = cv2.resize(img, target_size)
    
    # Apply preprocessing steps
    img_denoised = cv2.GaussianBlur(img_resized, (3, 3), 0)
    img_enhanced = cv2.equalizeHist(img_denoised)
    img_normalized = img_enhanced / 255.0
    
    return img_normalized

def process_image_batch(image_folder, pattern="*.jpg", max_images=None):
    """
    Process a batch of images from a folder
    """
    # Load models
    svm_hog, pca_model, svm_pca = load_models()
    if svm_hog is None:
        return
    
    # Get list of image files
    image_paths = glob.glob(os.path.join(image_folder, pattern))
    
    if not image_paths:
        print(f"No images found matching pattern '{pattern}' in folder '{image_folder}'")
        return
    
    # Limit number of images if specified
    if max_images is not None:
        image_paths = image_paths[:max_images]
    
    print(f"Processing {len(image_paths)} images...")
    
    # Create results directory if it doesn't exist
    results_dir = "batch_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Process each image
    results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # Load and preprocess image
        img = load_and_preprocess_image(image_path)
        if img is None:
            continue
        
        # Extract features
        img_hog = extract_hog_features(np.array([img]))
        img_pca = pca_model.transform(img.flatten().reshape(1, -1))
        
        # Make predictions
        hog_pred = svm_hog.predict(img_hog)[0]
        hog_prob = np.max(svm_hog.predict_proba(img_hog))
        
        pca_pred = svm_pca.predict(img_pca)[0]
        pca_prob = np.max(svm_pca.predict_proba(img_pca))
        
        # Store results
        results.append({
            'image_path': image_path,
            'hog_prediction': hog_pred,
            'hog_confidence': hog_prob,
            'pca_prediction': pca_pred,
            'pca_confidence': pca_prob
        })
        
        # Create visualization
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Image: {os.path.basename(image_path)}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.text(0.1, 0.7, f"HOG-SVM Prediction: Person {hog_pred}", fontsize=12)
        plt.text(0.1, 0.6, f"Confidence: {hog_prob:.2f}", fontsize=12)
        plt.text(0.1, 0.4, f"PCA-SVM Prediction: Person {pca_pred}", fontsize=12)
        plt.text(0.1, 0.3, f"Confidence: {pca_prob:.2f}", fontsize=12)
        
        plt.tight_layout()
        
        # Save visualization
        output_filename = os.path.join(results_dir, f"result_{os.path.basename(image_path)}.png")
        plt.savefig(output_filename)
        plt.close()
    
    # Create summary visualization
    if results:
        create_summary_visualization(results, results_dir)
    
    print(f"Batch processing complete. Results saved in '{results_dir}' directory.")

def create_summary_visualization(results, output_dir):
    """
    Create a summary visualization of batch processing results
    """
    # Count predictions by person ID
    hog_counts = {}
    pca_counts = {}
    
    for result in results:
        hog_pred = result['hog_prediction']
        pca_pred = result['pca_prediction']
        
        if hog_pred not in hog_counts:
            hog_counts[hog_pred] = 0
        hog_counts[hog_pred] += 1
        
        if pca_pred not in pca_counts:
            pca_counts[pca_pred] = 0
        pca_counts[pca_pred] += 1
    
    # Sort by person ID
    hog_ids = sorted(hog_counts.keys())
    pca_ids = sorted(pca_counts.keys())
    
    hog_values = [hog_counts[id] for id in hog_ids]
    pca_values = [pca_counts[id] for id in pca_ids]
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # HOG predictions
    plt.subplot(2, 1, 1)
    plt.bar(hog_ids, hog_values)
    plt.xlabel('Person ID')
    plt.ylabel('Count')
    plt.title('HOG-SVM Predictions')
    plt.xticks(hog_ids)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # PCA predictions
    plt.subplot(2, 1, 2)
    plt.bar(pca_ids, pca_values)
    plt.xlabel('Person ID')
    plt.ylabel('Count')
    plt.title('PCA-SVM Predictions')
    plt.xticks(pca_ids)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary.png'))
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process a batch of face images')
    parser.add_argument('image_folder', type=str, help='Path to folder containing face images')
    parser.add_argument('--pattern', type=str, default="*.jpg", help='File pattern to match (default: *.jpg)')
    parser.add_argument('--max', type=int, help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    process_image_batch(args.image_folder, args.pattern, args.max)
