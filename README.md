# Face Identification using HOG-PCA Feature Extraction and SVM Classifier

This project implements a face identification system based on the paper "Face Identification using HOG-PCA Feature Extraction and SVM Classifier". The system compares two feature extraction techniques (HOG and PCA) with SVM classification for face identification tasks.

## Overview

The face identification system follows these steps:
1. Load and preprocess face images
2. Extract features using HOG (Histogram of Oriented Gradients) and PCA (Principal Component Analysis)
3. Train SVM classifiers on both feature sets
4. Evaluate and compare the performance of both approaches

## Requirements

The following Python libraries are required:
- NumPy
- Matplotlib
- scikit-learn
- scikit-image
- OpenCV (cv2)

You can install these dependencies using pip:

```bash
pip install numpy matplotlib scikit-learn scikit-image opencv-python
```

## Dataset

The system uses the Olivetti Faces dataset, which contains 400 images of 40 distinct people (10 images per person). The images show faces with different lighting conditions, facial expressions, and facial details. The dataset is automatically downloaded by scikit-learn when running the code.

## Implementation Details

### 1. Image Preprocessing
- Noise reduction using Gaussian blur
- Image enhancement using histogram equalization
- Normalization to scale pixel values to [0, 1]

### 2. Feature Extraction
- **HOG (Histogram of Oriented Gradients)**: Extracts features based on the distribution of intensity gradients and edge directions
  - Parameters: 8x8 pixels per cell, 2x2 cells per block, 9 orientation bins
- **PCA (Principal Component Analysis)**: Reduces dimensionality by selecting the most important features
  - Uses 150 principal components as specified in the paper

### 3. Classification
- **SVM (Support Vector Machine)**: Trains classifiers on both HOG and PCA features
  - Uses grid search to find optimal hyperparameters (C, gamma, kernel)
  - Tests both linear and RBF kernels

### 4. Evaluation
- Compares HOG-SVM and PCA-SVM approaches using:
  - Accuracy and F1 score for performance metrics
  - Feature extraction time, training time, and prediction time for efficiency metrics

## How to Run

### Prior Statement
In the uploaded code, the plot.iqynb file finally fully implemented the requirements

### Training and Evaluation

To train the models and evaluate their performance on the Olivetti dataset:

```bash
python face_identification.py
```

### Training and Saving Models

To train models on the full dataset and save them for later use:

```bash
python save_trained_models.py
```

This script will train HOG-SVM and PCA-SVM models on the entire Olivetti dataset and save them to the 'models' directory.

### Testing with a Sample Image

To test the trained models with a random sample from the Olivetti dataset:

```bash
python test_with_sample.py
```

This script will select a random face image from the dataset, make predictions using both HOG-SVM and PCA-SVM models, and display the results.

### Using Your Own Images

To use the trained models with your own face images:

```bash
python predict_custom_image.py path/to/your/face_image.jpg
```

If you want to force retraining of the models:

```bash
python predict_custom_image.py path/to/your/face_image.jpg --train
```

Note: Custom images should be grayscale face images. The script will resize them to match the Olivetti dataset format (64x64 pixels).

### Real-time Webcam Face Identification

To run face identification in real-time using your webcam:

```bash
python webcam_face_identification.py
```

This script will:
1. Access your webcam
2. Detect faces in each frame using OpenCV's Haar Cascade classifier
3. Preprocess the detected face
4. Extract HOG and PCA features
5. Make predictions using both HOG-SVM and PCA-SVM models
6. Display the results in real-time

Press 'q' to quit the webcam application.

### Batch Processing Multiple Images

To process multiple face images at once:

```bash
python batch_process_images.py path/to/image/folder
```

Additional options:
- `--pattern`: File pattern to match (default: "*.jpg")
- `--max`: Maximum number of images to process

Example:
```bash
python batch_process_images.py my_faces --pattern "*.png" --max 10
```

This script will:
1. Process all matching images in the specified folder
2. Generate individual prediction visualizations for each image
3. Create a summary visualization showing the distribution of predictions
4. Save all results in the 'batch_results' directory

## Expected Results

The script will:
1. Load and preprocess the Olivetti faces dataset
2. Extract HOG and PCA features
3. Train SVM models on both feature sets
4. Evaluate and compare the models
5. Generate visualization files:
   - `face_identification_results.png`: Shows sample test images with true and predicted labels
   - `hog_pca_comparison.png`: Compares performance and time metrics between HOG and PCA approaches

## Results Interpretation

Based on the paper, we expect:
- PCA to be more computationally efficient than HOG for feature extraction
- HOG to potentially provide better accuracy for face identification
- The comparison will show trade-offs between accuracy and computational efficiency

The actual results may vary depending on the specific implementation and parameter settings.
