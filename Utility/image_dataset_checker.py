import os
import argparse
from PIL import Image
from tqdm import tqdm

def count_images_in_folder(folder_path):
    """
    Counts the number of images in a folder.

    Args:
    - folder_path: The path to the folder.

    Returns:
    - image_count: The number of images in the folder.
    """
    image_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is an image (you can customize the list of supported extensions)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_count += 1
    return image_count


def check_image_corruption(image_path, corrupted_images, verification_level):
    """
    Checks if an image is corrupted.

    Args:
    - image_path: The path to the image.
    - corrupted_images: A list to store corrupted image paths.
    - verification_level: The level of verification to perform on the image.

    Returns:
    - True if the image is not corrupted, False otherwise.
    """
    try:
        if verification_level == 1:
            Image.open(image_path).verify()
        elif verification_level == 2:
            im = Image.open(image_path)
            im.verify()
            im.close()
            im = Image.open(image_path)
            im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            im.close()
        return True  # Image is not corrupted
    except Exception as e:
        print(f"Corrupted image: {image_path} - {e}")
        corrupted_images.append(image_path)
        return False


def main(root_folder, verification_level):
    """
    The main function of the script.

    Args:
    - root_folder: The path to the root folder.
    - verification_level: The level of verification to perform on the images.
    """
    folder_count = 0
    total_images = 0
    corrupted_images = []

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            folder_count += 1
            image_count = count_images_in_folder(folder_path)
            total_images += image_count
            print(f"{folder_name}: {image_count} Images")

            # Check for image corruption in the folder
            for root, dirs, files in os.walk(folder_path):
                for file in tqdm(files):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        image_path = os.path.join(root, file)
                        check_image_corruption(image_path, corrupted_images, verification_level)

            # Display subfolder image counts
            for root, dirs, files in os.walk(folder_path):
                for subfolder_name in dirs:
                    subfolder_path = os.path.join(root, subfolder_name)
                    subfolder_image_count = count_images_in_folder(subfolder_path)
                    print(f"    {subfolder_name}: {subfolder_image_count} Images")

    print(f"\nTotal Images: {total_images} Images")
    print(f"Total Folders: {folder_count} Folders\n")

    if corrupted_images:
        print(f"Corrupted images count: {len(corrupted_images)} Images\n")
        print("Corrupted images paths:")
        for path in corrupted_images:
            print(path)

        delete_option = input("\nDo you want to delete the corrupted images? (y/N): ").lower()
        if delete_option == 'y':
            for path in corrupted_images:
                os.remove(path)
            print("Corrupted images deleted.")
        else:
            print("Corrupted images were not deleted.")
    else:
        print("No corrupted images found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Dataset Checker")
    parser.add_argument("root_folder", type=str, help="Path to the root folder")
    parser.add_argument("-v", "--verification_level", type=int, choices=[0, 1, 2], default=1,
                        help="Level of verification for image corruption (0, 1 or 2)")

    args = parser.parse_args()

    main(args.root_folder, args.verification_level)