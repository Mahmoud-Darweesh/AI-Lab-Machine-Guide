import os
import pickle
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from queue import Queue

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm


SCOPES = ['https://www.googleapis.com/auth/drive']
CREDS_FILE = 'credentials.json'  # Replace with your credentials file
DEFAULT_STORAGE_PATH = ''
GOOGLE_DRIVE_FOLDER_ID = ''


class FileDownloader:
    def __init__(self, destination_folder, folder_id):
        self.destination_folder = destination_folder
        self.folder_id = folder_id

    def authenticate(self) -> Credentials:
        """
        Authenticates the user and returns the credentials.
        """
        flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
        credentials = flow.run_local_server(port=0)
        return credentials

    def list_files(self, service, folder_id, destination_folder):
        """
        Lists all files within a specific folder in Google Drive.

        Args:
        - service: The authenticated Google Drive service.
        - destination_folder: The destination folder path to store the downloaded files.

        Returns:
        - A list of files.
        """
        all_files = []
        page_token = None

        while True:
            results = service.files().list(
                q=f"'{folder_id}' in parents",
                fields="nextPageToken, files(id, name, mimeType)",
                pageSize=1000,
                pageToken=page_token
            ).execute()

            files = results.get('files', [])
            for file in files:
                item_id = file['id']
                item_name = file['name']
                item_type = file['mimeType']
                file_path = os.path.join(destination_folder, item_name)

                if item_type == 'application/vnd.google-apps.folder':
                    all_files.extend(self.list_files(service, item_id, file_path))
                else:
                    all_files.append({"item_id": item_id, "file_path": file_path})

            page_token = results.get('nextPageToken')
            if not page_token:
                break

        return all_files

    def download_file(self, service, file_id, file_path):
        """
        Downloads a specific file from Google Drive.

        Args:
        - service: The authenticated Google Drive service.
        - file_id: The ID of the file.
        - file_path: The file path to save the downloaded file.

        Returns:
        - The file path of the downloaded file.
        """
        request = service.files().get_media(fileId=file_id)

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            downloader = MediaIoBaseDownload(file, request)
            done = False
            while not done:
                _, done = downloader.next_chunk(num_retries=2)

        return file_path

    def download_file_wrapper(self, args):
        """
        Wrapper function for the download_file method to be used with multiprocessing.

        Args:
        - args: Arguments tuple containing the service and file information.
        """
        service, file_info = args
        file_id = file_info['item_id']
        file_path = file_info['file_path']
        self.download_file(service, file_id, file_path)

    def main(self, processes):
        """
        Main function to authenticate, create the Drive API service, list files in the folder, and download the files.
        """
        # Authenticate and create the Drive API service
        credentials = None
        if os.path.exists('token.pickle'):
            # Read the token from the file and
            # store it in the variable self.creds
            with open('token.pickle', 'rb') as token:
                credentials = pickle.load(token)

        if not credentials or not credentials.valid:
            credentials = self.authenticate()
            with open('token.pickle', 'wb') as token:
                pickle.dump(credentials, token)

        service = build('drive', 'v3', credentials=credentials)

        # List all files in the folder
        files = self.list_files(service, self.folder_id, self.destination_folder)

        start_time = time.time()

        # Download each file with tqdm progress bar using multiprocessing
        with Pool(processes=processes) as pool:
            with tqdm(total=len(files), desc='Downloading files', unit='file') as pbar:
                for _ in pool.imap_unordered(self.download_file_wrapper, [(service, file_info) for file_info in files]):
                    pbar.update()

        end_time = time.time()
        total_time = end_time - start_time

        print(f'Download completed successfully in {total_time:.2f} seconds.')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Google Drive File Downloader')
    parser.add_argument('-d', '--destination', type=str, default='downloads', help='Destination folder path to store downloaded files')
    parser.add_argument('-f', '--folder-id', type=str, help='ID of the Google Drive folder', required=True)
    parser.add_argument('-p', '--processes', type=int, help='number of workers used to download the folder', required=True)
    args = parser.parse_args()

    destination_folder = args.destination
    folder_id = args.folder_id
    workers = args.processes

    downloader = FileDownloader(destination_folder, folder_id)
    downloader.main(workers)