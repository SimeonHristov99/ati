""" This module implements the logic for gathering the texts. """

from dataclasses import dataclass
from typing import List

import httpx
import pandas as pd
from selectolax.parser import HTMLParser


@dataclass
class Text:
    """This class represents each row in the initial dataframe."""
    author: str
    title: str
    download_link: str


def get_html(url: str) -> HTMLParser:
    """
    Parse a response from a GET request.

    Args:
        url (str): Full URL for the path.

    Returns:
        HTMLParser: The text of the parsed response.
    """
    resp = httpx.get(url)
    html_parsed = HTMLParser(resp.text)
    return html_parsed


def parse_texts(author: str, url: str, html: HTMLParser) -> List[Text]:
    """
    For each author return all their texts.

    Args:
        author (str): Name of the author.
        url (str): Full URL from which texts can be found and downloaded.
        html (HTMLParser): Parsed HTML.

    Returns:
        List[Text]: The texts of the author.
    """
    items = html.css('dl.text-entity')

    results = [
        Text(
            author=author,
            title=item.css_first('a.textlink').text().strip(" „“…!?"),
            download_link=url + item.css_first(
                'a.dl-txt').attributes.get('href').strip(),
        ) for item in items
    ]

    return results


def scrape(url: str, authors: List[str]) -> pd.DataFrame:
    """
    Returns a dataframe with the name of the author, title of text written
    by them and a link from it can be downloaded.

    Args:
        url (str): The full URL which will be used to download the texts.
            It must not end with a '\'!
        authors (List[str]): The authors for which to extract the texts.

    Returns:
        pd.DataFrame: A dataframe with the name of the author, title of text
        written by them and a link from it can be downloaded.
    """
    if url[-1] == '\\':
        raise ValueError("Url must not end with a '\\' character!")

    download_url = url.removesuffix('/person')

    all_texts = []
    for author in authors:
        texts_url = f'{url}/{author}#texts'
        html = get_html(texts_url)
        texts = parse_texts(author, download_url, html)
        all_texts.extend(texts)

    result = pd.DataFrame(all_texts)
    return result


if __name__ == '__main__':
    print('Hello from scraper.py!')
