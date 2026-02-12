# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Plant disease detection web app using a custom CNN (PyTorch) served via Flask. Users upload leaf images, the model classifies into one of 39 classes (diseases/healthy states across 14 plant species), and the app returns disease info with recommended supplements.

## Setup & Run

```bash
# Requires Python 3.8
py -3.8 -m venv venv
venv\Scripts\activate
pip install -r "Flask Deployed App\requirements.txt"

# Download model file (not in repo, .pt files are gitignored):
# plant_disease_model_1_latest.pt from Google Drive link in README
# Place it in "Flask Deployed App/"

# Run the Flask app
cd "Flask Deployed App"
python app.py
# Serves at http://127.0.0.1:5000/
```

Production deployment uses gunicorn via `Procfile`: `web: gunicorn app:app`

## Architecture

### CNN Model (`Flask Deployed App/CNN.py`)
- Custom 4-block CNN: each block is Conv2d -> ReLU -> BatchNorm2d -> Conv2d -> ReLU -> BatchNorm2d -> MaxPool2d
- Channel progression: 3 -> 32 -> 64 -> 128 -> 256
- Dense layers: Flatten(50176) -> Dropout(0.4) -> Linear(1024) -> ReLU -> Dropout(0.4) -> Linear(K)
- K=39 output classes, expects 224x224 RGB input
- `idx_to_classes` dict maps prediction indices to class names (e.g., `0: 'Apple___Apple_scab'`)

### Flask App (`Flask Deployed App/app.py`)
- Loads model at startup with `torch.load("plant_disease_model_1_latest.pt")` and sets to eval mode
- Reads `disease_info.csv` and `supplement_info.csv` (cp1252 encoding) into pandas DataFrames at startup
- **Routes:**
  - `/`, `/index` - Upload page (index.html)
  - `/submit` (POST) - Accepts image upload, saves to `static/uploads/`, runs prediction, renders results
  - `/market` - Supplement marketplace listing all products
- **Prediction pipeline:** PIL Image -> resize(224,224) -> TF.to_tensor -> model forward pass -> argmax -> index into CSV DataFrames

### Templates (Jinja2, extend `base.html`)
- `base.html` - Layout with navbar, Bootstrap 5, Font Awesome, Poppins font, green theme (#4ECC5A)
- `index.html` - File upload with drag-and-drop and camera capture via JS
- `submit.html` - Results page; uses hardcoded healthy-plant indices (3,5,7,11,15,18,20,23,24,25,28,38) to toggle between "healthy" and "disease" display modes
- `market.html` - Product grid with client-side filtering (fertilizer/supplement categories)
- `home.html` - Legacy landing page (not routed in current app.py)

### Data Files
- `disease_info.csv` - 39 rows: disease_name, description, prevention steps, reference image URL
- `supplement_info.csv` - 39 rows: supplement name, image URL, buy link (index 4 "Background_without_leaves" has empty supplement data)

### Model Training (`Model/`)
- Jupyter notebook trained on Plant Village Dataset (61,486 images)
- Train/validation/test split: 85%/15% with 70/30 sub-split on train portion
- Batch size 64, Adam optimizer, CrossEntropyLoss, trained for 5 epochs
- Reported accuracy: ~97% train, ~99% test/validation

## Key Conventions

- Uploaded images are saved to `Flask Deployed App/static/uploads/` (this directory must exist)
- The model file name in app.py is `plant_disease_model_1_latest.pt` (differs from README which references `plant_disease_model_1.pt`)
- CSV files use `cp1252` encoding, not UTF-8
- Healthy plant indices are hardcoded in templates for conditional rendering - if class ordering changes, these must be updated in both `submit.html` and `market.html`
- The app runs on CPU only (PyTorch CPU-only wheels in requirements.txt)
- Dependencies are pinned to older versions (Flask 1.1.2, PyTorch 1.8.1+cpu, Pillow 8.2.0)
