#!/bin/bash
# Download the CLARE dataset from the Borealis Data repository.
# DOI: 10.5683/SP3/H0AELT
#
# The dataset is hosted at: https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/H0AELT
#
# Manual steps (recommended):
#   1. Visit the URL above in your browser
#   2. Click "Access Dataset" → Download ZIP
#   3. Extract into ./data/
#
# The expected layout after extraction:
#   data/
#   ├── ECG/    P01/ ... P24/
#   ├── EDA/    P01/ ... P24/
#   ├── EEG/    P01/ ... P24/
#   ├── Gaze/   P01/ ... P24/
#   └── Labels/ P01/ ... P24/
#
# Alternatively, clone the GitHub repo (may contain only a subset):
#   git clone https://github.com/Prithila05/CLARE.git data

echo "Please download the CLARE dataset manually from:"
echo "  https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/H0AELT"
echo ""
echo "Extract it into ./data/ so the structure is:"
echo "  data/ECG/   data/EDA/   data/EEG/   data/Gaze/   data/Labels/"
