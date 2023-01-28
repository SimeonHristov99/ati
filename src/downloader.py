""" Populates the `./DATA/original` directory with text files. """

import re
import zipfile
from pathlib import Path
from typing import List

import requests


def download_file(url: str, output_dir: str) -> str:
    """Downloads a file from the `url` and saves the contents in `local_filename`.

    Args:
        url (str): The full url to the resource to be downloaded.
        output_dir (str): A (relative or absolute) path to
            the directory in which the file will be saved.

    Returns:
        str: The name of the file created on the system that holds the file.
    """
    local_filename = f"{output_dir}/{url.split('/')[-1]}"
    response = requests.get(url, timeout=5)

    if response is None:
        raise TimeoutError(f'Could not reach {url}')

    if response.status_code != 200:
        raise ValueError(f'Incorrect URL: {url}')

    with open(local_filename, 'wb') as resp:
        resp.write(response.content)
    return local_filename


def unzip_file(filename: str, output_dir: str) -> str:
    """Unzips the first file in a zip archive.

    Args:
        filename (str): Relative path to the zip file.
        output_dir (str): Directory in which the first file will be unzipped.

    Returns:
        str: The name of the unzipped file.
    """
    zipdata = zipfile.ZipFile(filename)
    zipinfo = zipdata.infolist()[0]

    name = ''.join(zipinfo.filename.split('.')[:-1])
    zipinfo.filename = re.sub(r"[^a-zA-Z]{2}|\d", "", name.lower())
    zipinfo.filename = re.sub(r"\s", "_", zipinfo.filename) + '.txt'

    zipdata.extract(zipinfo, path=output_dir)
    return zipinfo.filename


def remove_zipfiles(path: str) -> str:
    """Removes all zipfiles in a directory. Not recursive.

    Args:
        path (str): Relative or absolute path to the directory.

    Returns:
        str: The name of the directory.
    """
    directory = Path(path)
    for file in directory.glob('*.zip'):
        file.unlink()
    return directory.name


def download_and_extract_zips(urls: List[str], output_dir: str) -> List[str]:
    """Downloads a zip file into a specified directory
        and extracts its contents.

    Args:
        urls (List[str]): The urls from which to download.
        output_dir (str): The place where the files will be saved and extracted.
        
    Returns:
        List[str]: The names of the extracted text files.
    """
    filenames = []
    for i, url in enumerate(urls):
        filename = download_file(url, output_dir)
        unzipped_filename = unzip_file(filename, output_dir)
        unzipped_filename = f'{output_dir}/{unzipped_filename}'
        filenames.append(unzipped_filename)
        print(f'Extracted {i+1:02}: {unzipped_filename}')

    remove_zipfiles(output_dir)
    return filenames


if __name__ == '__main__':
    print('Hello, from downloader.py!')
