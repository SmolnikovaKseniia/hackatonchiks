import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import VGG16 
import shutil

# Load pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Function to extract features from image using VGG16
def extract_features(image_path, model):
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    features = features.flatten()  # Flatten the feature array
    return features

# Function to perform clustering and organize images into folders
def cluster_faces(image_folder, output_folder):
    # Define the number of clusters
    num_clusters = 10  # Limit the number of clusters to 10

    # Initialize KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # Lists to store image paths and corresponding feature vectors
    image_paths = []
    features = []

    # Iterate through image folder to extract features
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            image_paths.append(image_path)
            feature = extract_features(image_path, model)
            features.append(feature)

    # Convert lists to numpy arrays
    features = np.array(features)

    # Adjust the number of components for PCA
    num_components = min(min(features.shape), 100)  # Use the minimum of samples or features or 100
    pca = PCA(n_components=num_components, random_state=42)

    # Reduce dimensionality of feature vectors using PCA
    features_reduced = pca.fit_transform(features)

    # Perform KMeans clustering
    kmeans.fit(features_reduced)

    # Create output folders for each cluster
    for i in range(num_clusters):
        cluster_folder = os.path.join(output_folder, f'Cluster_{i}')
        os.makedirs(cluster_folder, exist_ok=True)

    # Move images to corresponding cluster folders
    for image_path, label in zip(image_paths, kmeans.labels_):
        cluster_folder = os.path.join(output_folder, f'Cluster_{label}')
        shutil.copy(image_path, cluster_folder)

# Define input and output folders
input_folder = "C:\\Users\\Ksena\\Documents\\kpi\\zhostka\\hackatonchiks\\result_folder"
output_folder = "C:\\Users\\Ksena\\Documents\\kpi\\zhostka\\hackatonchiks\\clustered_images"

# Perform face clustering and organize images into folders
cluster_faces(input_folder, output_folder)



