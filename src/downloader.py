""" Populates the `./DATA/original` directory with text files. """

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


def main():
    """Main function for testing purposes.
    """
    URL = '	https://chitanka.info/text/5283-andreshko.txt.zip'
    output_dir = './DATA/original'
    loc = download_file(URL, output_dir)
    print(f'Saved to {loc}')


if __name__ == '__main__':
    main()
