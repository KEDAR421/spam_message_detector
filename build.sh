#!/bin/bash

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data to a specific directory
python -m nltk.downloader -d /opt/render/project/src/.nltk_data punkt stopwords
