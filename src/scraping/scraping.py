import os

import requests
from bs4 import BeautifulSoup
import pandas as pd
import spacy
import pycountry
import nltk
from nltk.tokenize import sent_tokenize
import time
import openai
from getpass import getpass

# Set up OpenAI API client
if not os.environ.get('OPENAI_API_KEY'):
    api_key = getpass("Enter your OpenAI API key: ")
else:
    api_key = os.environ['OPENAI_API_KEY']
client = openai.OpenAI(api_key=api_key)
# Ensure required NLTK data is downloaded
nltk.download('punkt')

# Load spaCy English model (download with: python -m spacy download en_core_web_sm)
nlp = spacy.load('en_core_web_sm')

BASE_URL = 'https://en.wikipedia.org'
LIST_URL = f"{BASE_URL}/wiki/Wikipedia:List_of_controversial_issues#Religion"

def summarize_text(text):
    """Generate a 2–3 line summary of the given text using OpenAI."""
    prompt = f"Summarize the following text in 2–3 lines:\n\n{text}"
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def fetch_first_paragraph(title: str):
    """
    Fetch the lead paragraph (plain-text extract) of a Wikipedia page
    via the REST API.
    """
    endpoint = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + title.replace(" ", "_")
    )
    resp = requests.get(endpoint, headers={"Accept": "application/json"})
    if resp.status_code != 200:
        return None

    data = resp.json()
    # `extract` is the first paragraph(s) as plain text
    return data.get("extract")


def extract_countries(text):
    """Use spaCy NER to identify country names in text and validate with pycountry."""
    doc = nlp(text)
    gpes = set(ent.text for ent in doc.ents if ent.label_ == 'GPE')
    countries = []
    for name in gpes:
        try:
            country = pycountry.countries.lookup(name)
            countries.append(country.name)
        except LookupError:
            continue
    return sorted(set(countries))


def fetch_official_languages(country_name):
    """Fetch official languages for a country from its Wikipedia infobox."""
    try:
        page = requests.get(f"{BASE_URL}/wiki/{country_name.replace(' ', '_')}")
        soup = BeautifulSoup(page.text, 'html.parser')
        infobox = soup.find('table', class_='infobox')
        langs = []
        if infobox:
            for row in infobox.find_all('tr'):
                header = row.find('th')
                if header and 'Official' in header.text:
                    cell = row.find('td')
                    # split on line breaks and commas
                    for part in cell.get_text(separator=',').split(','):
                        lang = part.strip()
                        if lang:
                            langs.append(lang)
                    break
        return langs
    except Exception:
        return []


def main():
    # Step 1: Parse the list page
    resp = requests.get(LIST_URL)
    soup = BeautifulSoup(resp.text, 'html.parser')

    # Locate the "Politics and economics" section
    section = soup.find(id='Religion')
    ul = section.find_next('ul')
    items = ul.find_all('li')

    records = []

    for li in items:
        print(f"Processing item: {li.text}")
        link = li.select_one('a[href^="/wiki/"]:not([href*=":"])')
        if not link:
            print(f"Did not find a valid link in {li.text}")
            continue
        topic_title = link.get('title')

        # Fetch and summarize first paragraph
        para = fetch_first_paragraph(topic_title)
        if not para:
            print(f"Failed to fetch paragraph for {topic_title}")
            continue
        sentences = sent_tokenize(para)
        summary = summarize_text("".join(sentences))
        print(f"Processing: {topic_title}")
        print(f"Summary: {summary}")
        # Extract countries and languages
        countries = extract_countries(para)
        languages = []
        for country in countries:
            langs = fetch_official_languages(country)
            for lang in langs:
                if lang not in languages:
                    languages.append(lang)
            if len(languages) >= 2:
                break

        # Ensure exactly two language entries
        languages += [''] * (2 - len(languages))

        records.append({
            'statement': summary,
            'countries': ";".join(countries),
            'language1': languages[0],
            'language2': languages[1]
        })

        # Be polite to Wikipedia
        time.sleep(1)

    # Write to CSV
    df = pd.DataFrame.from_records(records, columns=['statement', 'countries', 'language1', 'language2'])
    df.to_csv('controversial_religion.csv', index=False)
    print("CSV file generated: controversial_religion.csv")


if __name__ == "__main__":
    main()
