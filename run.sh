#!/bin/bash

# Install packages from requirements.txt
pip3 install -r requirements.txt

# Download TextBlob corpora
python3 -m textblob.download_corpora

# Download spaCy English model
python3 -m spacy download en_core_web_sm
