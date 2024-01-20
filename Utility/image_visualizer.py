"""
PCA Code
Name: Mahmoud Darwish
ID: b00095043

Info:

t-SNE and PCA Image Visualization

This Python script performs dimensionality reduction using t-SNE or PCA on a collection of images and generates visualizations in 2D. It loads images from specified folders, extracts features, applies dimensionality reduction techniques, and visualizes the results. The script provides options to save the reduced features as a pickle file and to generate scatter plots with or without annotations.

Installation

To use the t-SNE and PCA Image Visualization script, you need to have the following dependencies installed:

pip install numpy torch torchvision Pillow scipy tqdm

Usage

python image_visualizer.py --datasets folder1 folder2 folder3 ... --method <tsne or pca> [--save-features] [--features-path path/to/features.pickle] [--visualization-path path/to/visualization.png]

Arguments

- `--datasets`: List of paths to the folders containing image datasets.
- `--batch-size`: Batch size for image processing.
- `--method`: Dimensionality reduction method to use (tsne or pca).
- `--save-features`: Optional flag to save extracted features as a pickle file.
- `--features-path`: Path to save or load the features pickle file (default: features.pickle).
- `--visualization-path`: Path to save the visualization plot (default: None).

Examples

python image_visualizer.py --datasets dataset1 dataset2 --method tsne --save-features --visualization-path visualization.png

python image_visualizer.py --datasets /main/dataset/folder --method pca

Note

- Supported image formats: PNG, JPEG, GIF, BMP.
- The script generates visualizations and allows annotations using t-SNE or PCA.
- The `--save-features` option enables saving the extracted features for future use.
- Specify either `tsne` or `pca` as the `method` argument to choose the dimensionality reduction technique.

"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
from tqdm import tqdm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def load_image_paths_from_folder(folder_path):
    """
    Load images from a specified folder path.
    Args:
        folder_path (str): Path to the folder containing images.
    Returns:
        list: List of image paths.
    """
    images_paths = []
    for foldername, _, filenames in os.walk(folder_path):
        print("Checking path:", foldername)
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(foldername, filename)
                try:
                    with Image.open(image_path):
                        images_paths.append(image_path)
                except (IOError, SyntaxError):
                    print(f"Warning: Skipped corrupted image file: {image_path}")
    return images_paths


def save_features_to_pickle(features, file_path):
    """
    Save extracted features to a pickle file.
    Args:
        features (list): List of feature vectors.
        file_path (str): Path to save the pickle file.
    """
    print("Saving extracted features to a pickle file at:", file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(features, f)


def load_features_from_pickle(file_path):
    """
    Load features from a pickle file.
    Args:
        file_path (str): Path to the pickle file.
    Returns:
        list: List of feature vectors.
    """
    print("Loading extracted features from a pickle file at:", file_path)
    with open(file_path, 'rb') as f:
        features = pickle.load(f)
    return features


def apply_dimensionality_reduction(features, method='tsne', n_components=2, random_state=420):
    """
    Apply dimensionality reduction on the given feature vectors.
    Args:
        features (list): List of feature vectors.
        method (str): Dimensionality reduction method to use ('tsne' or 'pca').
        n_components (int): Number of components in the reduced-dimensional representation.
        random_state (int): Random state for reproducibility.
    Returns:
        np.ndarray: Reduced-dimensional embeddings.
    """
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state)
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    else:
        raise ValueError("Invalid dimensionality reduction method. Choose 'tsne' or 'pca'.")

    embeddings = reducer.fit_transform(np.asarray(features, dtype=np.float32))
    return embeddings


def visualize_embeddings(embeddings, classes, save_path=None):
    """
    Visualize embeddings using a scatter plot with different colors for different classes.
    Args:
        embeddings (np.ndarray): Embeddings.
        classes (list): List containing string class labels for each data point.
        save_path (str): Path to save the visualization plot (optional).
    """
    print("Visualizing embeddings using a scatter plot...")
    plt.figure()
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=classes, palette='viridis')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def annotate_plot_with_images(features, image_paths, classes, save_path=None):
    """
    Annotate scatter plot with images and color the points based on different classes.
    Args:
        features (list): List of feature vectors.
        image_paths (list): List of original image paths.
        classes (list): List of class labels corresponding to each data point.
        save_path (str): Path to save the annotated plot.
    """
    plt.figure(figsize=(16, 12))
    ax = plt.subplot(111)
    with tqdm(total=len(image_paths), desc="Annotating scatter plot with images...") as pbar:
        for i, (x, y) in enumerate(features):
            img_path = image_paths[i]
            img = Image.open(img_path).resize((30, 30))
            imagebox = OffsetImage(img, zoom=0.5)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0)
            ax.add_artist(ab)
            pbar.update(1)
    
    features = np.array(features)
    ax.set_xlim(min(features[:, 0]) - 5, max(features[:, 0]) + 5)
    ax.set_ylim(min(features[:, 1]) - 5, max(features[:, 1]) + 5)
    ax.set_title('Visualization with Images')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    # Add color based on classes
    unique_classes = list(set(classes))
    colors = plt.cm.get_cmap('viridis', len(unique_classes))
    for i, class_label in enumerate(unique_classes):
        class_indices = [j for j in range(len(classes)) if classes[j] == class_label]
        ax.scatter(features[class_indices, 0], features[class_indices, 1], color=colors(i), label=class_label)

    ax.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def image_generator(folder_paths, batch_size=10):
    """
    A generator that yields flattened image vectors from specified image folders in batches.
    Args:
        folder_paths (list): List of paths to the folders containing images.
        batch_size (int, optional): Number of images to process per batch. Defaults to 10.
    Yields:
        list: List of flattened image vectors in each batch.
            Each element represents a flattened image as a 1-dimensional numpy array.
            Skips corrupted image files and prints a warning for each skipped file.
    """
    image_paths = []

    for folder_path in folder_paths:
        print("Checking images from:", folder_path)
        image_paths.extend(load_image_paths_from_folder(folder_path))
        print("Done")

    flattened_images = []
    class_names = []
    with tqdm(total=len(image_paths), desc=("Loading Images... (0/" + str(int(len(image_paths)/batch_size)) + ")")) as pbar:
        for i, path in enumerate(image_paths):
            try:
                with Image.open(path) as img:
                    img = img.resize((64, 64), Image.Resampling.LANCZOS)
                    img_numpy_array = np.asarray(img).flatten()
                    flattened_images.append(img_numpy_array)
                    class_names.append(os.path.basename(os.path.dirname(path)))
                    pbar.update(1)
                if len(flattened_images) >= batch_size and len(image_paths) - i >= batch_size:
                    pbar.set_description("Running analysis...")
                    yield flattened_images, class_names
                    flattened_images = []  # Clear for memory space
                    class_names = []
                    pbar.set_description("Loading Images... (" + str(int((i + 1) / batch_size)) + "/" + str(int(len(image_paths)/batch_size)) + ")")
            except (IOError, SyntaxError):
                print(f"Warning: Skipped corrupted image file (during generation): {path}")
        if flattened_images:
            pbar.set_description("Running analysis...")
            yield flattened_images, class_names
            flattened_images = []
            class_names = []



def main(args):
    classes = []
    features = []

    generator = image_generator(args.datasets, args.batch_size)

    for batch_images, class_names in generator:
        embeddings = apply_dimensionality_reduction(batch_images, args.method)
        features.extend(embeddings)
        classes.extend(class_names)
        batch_images = []
        class_names = []

    if args.save_features:
        save_features_to_pickle(features, args.features_path)
        features_test = load_features_from_pickle(args.features_path)
    else:
        features_test = features

    visualize_embeddings(np.array(features_test), classes, args.visualization_path)

    image_paths = []

    for folder_path in args.datasets:
        print("Checking images from:", folder_path)
        image_paths.extend(load_image_paths_from_folder(folder_path))
        print("Done")
    
    annotate_plot_with_images(features_test, image_paths, classes, args.visualization_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='t-SNE and PCA Image Visualization')
    parser.add_argument('--datasets', nargs='+', help='List of dataset folders')
    parser.add_argument('--batch-size', type=int, default=20000, help='Batch size for image processing')
    parser.add_argument('--method', type=str, choices=['tsne', 'pca'], default='tsne', help='Dimensionality reduction method')
    parser.add_argument('--save-features', action='store_true', help='Save extracted features to a pickle file')
    parser.add_argument('--features-path', type=str, default='features.pickle', help='Path to save/load the features pickle file')
    parser.add_argument('--visualization-path', type=str, default=None, help='Path to save the visualization plot')

    args = parser.parse_args()
    main(args)