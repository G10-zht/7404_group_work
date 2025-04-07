from face_identification import load_olivetti_dataset

# Test the load_olivetti_dataset function
X, y, X_images = load_olivetti_dataset()
print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Number of unique person IDs: {len(set(y))}")
print(f"Person IDs: {sorted(set(y))}")
print(f"X_images shape: {X_images.shape}")
