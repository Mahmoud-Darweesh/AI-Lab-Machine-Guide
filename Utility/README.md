### Image Dataset Checker

This [script](/Utility/image_dataset_checker.py) is designed to check an image dataset and provide information about it. It counts the number of images in each folder of a given root folder and checks for image corruption. It also offers the option to delete corrupted images.

#### Installation

To use the Google Drive File Downloader script, you need to install the following dependencies:

```sh
pip install Pillow argparse tqdm
```

#### Usage

```sh
python image_dataset_checker.py <root_folder> [-v <verification_level>]
```

#### Arguments

- `root_folder`: The path to the root folder containing the image dataset.
- `-v` or `--verification_level`: (Optional) The level of verification to perform on the images. Default is 1:
  - `0`: Preform no verification
  - `1`: Perform basic verification using `Image.open().verify()`.
  - `2`: Perform advanced verification by additionally flipping the image horizontally.

#### Example Usage

1. Check image dataset in a folder:
   ```sh
   python image_dataset_checker.py /path/to/root/folder
   ```

2. Check image dataset and perform advanced verification:
   ```sh
   python image_dataset_checker.py /path/to/root/folder -v 2
   ```

### Google Drive File Downloader

This [script](/Utility/Google%20Drive%20Utils/google_drive_folder_downloader.py) allows users to easily download files from their Google Drive. It is designed to be used in a world-class AI Lab environment and is suitable for a public repository on GitHub.

#### Installation

To use the Google Drive File Downloader script, you need to install the following dependencies:

```sh
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib tqdm argparse
```

#### Usage

```sh
python google_drive_folder_downloader.py [-d <destination_folder>] [-f <folder_id>]
```

#### Arguments

- `-d` or `--destination`: (Optional) The destination folder path to store the downloaded files. By default, it is set to 'downloads'.
- `-f` or `--folder-id`: The ID of the Google Drive folder from which to download files.
- `-p` or `--processes`: The number of workers used to install the folder. The more you add the faster it will download, but the more cpu usage it will take.

#### Example Usage

1. Download files from a specific Google Drive folder:
   ```sh
   python google_drive_folder_downloader.py -d /path/to/destination/folder -f <google_drive_folder_id> -p 40
   ```

2. Download files from the default Google Drive folder and store them in the default 'downloads' folder:
   ```sh
   python google_drive_folder_downloader.py -f <google_drive_folder_id> -p 40
   ```

### FID Score

This [script](/Utility/FID%20Score/fid_score.py) allows users to easily calculate the FID Score between 2 datasets.

#### Installation

To use the FID Score script, you need to install the following dependencies:

```sh
pip install numpy torch torchvision Pillow scipy tqdm
```

#### Usage

```sh
python fid_score.py "/path/to/dataset1" "/path/to/dataset2" --device cuda:0
```

#### Credits

The owner of this script and the original script can be found [here](https://github.com/mseitzer/pytorch-fid/tree/master).

### t-SNE and PCA Image Visualization

This [script](/Utility/image_visualizer.py) performs dimensionality reduction using t-SNE or PCA on a collection of images and generates visualizations in 2D. It loads images from specified folders, extracts features, applies dimensionality reduction techniques, and visualizes the results. The script provides options to save the reduced features as a pickle file and to generate scatter plots with or without annotations.

#### Installation

To use the t-SNE and PCA Image Visualization script, you need to have the following dependencies installed:

```sh
pip install numpy torch torchvision Pillow scipy tqdm
```

#### Usage

```sh
python image_visualizer.py --datasets folder1 folder2 folder3 ... --method <tsne or pca> [--save-features] [--features-path path/to/features.pickle] [--visualization-path path/to/visualization.png]
```

#### Arguments

- `--datasets`: List of paths to the folders containing image datasets.
- `--batch-size`: Batch size for image processing.
- `--method`: Dimensionality reduction method to use (tsne or pca).
- `--save-features`: Optional flag to save extracted features as a pickle file.
- `--features-path`: Path to save or load the features pickle file (default: features.pickle).
- `--visualization-path`: Path to save the visualization plot (default: None).

#### Example Usage

```sh
python image_visualizer.py --datasets dataset1 dataset2 --method tsne --save-features --visualization-path visualization.png
```

```sh
python image_visualizer.py --datasets /main/dataset/folder --method pca
```

#### Note

- Supported image formats: PNG, JPEG, GIF, BMP.
- The script generates visualizations and allows annotations using t-SNE or PCA.
- The `--save-features` option enables saving the extracted features for future use.
- Specify either `tsne` or `pca` as the `method` argument to choose the dimensionality reduction technique.