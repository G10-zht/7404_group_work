import numpy as np
import cv2
import pickle
import os
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

def preprocess_frame(frame, face_cascade, target_size=(64, 64)):
    """
    Detect and preprocess a face from a webcam frame
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return None, None
    
    # Process the largest face (assuming it's the main subject)
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    
    # Extract and preprocess the face
    face_img = gray[y:y+h, x:x+w]
    face_img_resized = cv2.resize(face_img, target_size)
    
    # Apply preprocessing steps
    face_img_denoised = cv2.GaussianBlur(face_img_resized, (3, 3), 0)
    face_img_enhanced = cv2.equalizeHist(face_img_denoised)
    face_img_normalized = face_img_enhanced / 255.0
    
    return face_img_normalized, largest_face

def run_webcam_identification():
    """
    Run face identification on webcam feed
    """
    # Load models
    svm_hog, pca_model, svm_pca = load_models()
    if svm_hog is None:
        return
    
    # Load face cascade
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Start webcam
    print("Starting webcam. Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Preprocess frame
        face_img, face_rect = preprocess_frame(frame, face_cascade)
        
        if face_img is not None:
            # Extract features
            face_hog = extract_hog_features(np.array([face_img]))
            face_pca = pca_model.transform(face_img.flatten().reshape(1, -1))
            
            # Make predictions
            hog_pred = svm_hog.predict(face_hog)[0]
            hog_prob = np.max(svm_hog.predict_proba(face_hog))
            
            pca_pred = svm_pca.predict(face_pca)[0]
            pca_prob = np.max(svm_pca.predict_proba(face_pca))
            
            # Draw rectangle around face
            x, y, w, h = face_rect
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display predictions
            hog_text = f"HOG: Person {hog_pred} ({hog_prob:.2f})"
            pca_text = f"PCA: Person {pca_pred} ({pca_prob:.2f})"
            
            cv2.putText(display_frame, hog_text, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, pca_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Face Identification', display_frame)
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

if __name__ == "__main__":
    run_webcam_identification()
