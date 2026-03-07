# Plant Disease Detection App - Setup Guide

Follow these steps one by one to run the project on your computer.

---

## Step 1: Install Python

Download and install **Python 3.10 or above** from:
https://www.python.org/downloads/

> **Important:** During installation, check the box that says **"Add Python to PATH"**.

To verify Python is installed, open **Command Prompt** (search "cmd" in Start menu) and type:

```
python --version
```

You should see something like `Python 3.12.x`. If you see an error, restart your computer and try again.

---

## Step 2: Download or Clone the Project

**Option A - Download ZIP:**
1. Go to the GitHub repository page
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Extract the ZIP to a folder (e.g., `C:\plant-project`)

**Option B - Clone with Git:**
```
git clone https://github.com/AVISHKAR-PROJECTS-HACKNCRAFTS/-Advanced-Deep-Neural-Network-Framework-for-Early-Plant-Disease-Identification-and-Precision-Farming.git
```

---

## Step 3: Open Command Prompt in the Project Folder

1. Open **Command Prompt** (search "cmd" in Start menu)
2. Navigate to the project folder. Type:

```
cd "C:\plant-project\-Advanced-Deep-Neural-Network-Framework-for-Early-Plant-Disease-Identification-and-Precision-Farming"
```

> Replace the path above with wherever you saved the project.

---

## Step 4: Create a Virtual Environment

Because the project folder name is very long, create the virtual environment in a short path:

```
python -m venv C:\plant-venv
```

Now activate it:

```
C:\plant-venv\Scripts\activate
```

You should see `(plant-venv)` appear at the beginning of your command line. This means the virtual environment is active.

---

## Step 5: Install Required Packages

```
pip install -r "Flask Deployed App\requirements.txt"
```

This will download and install all necessary libraries. It may take a few minutes.

---

## Step 6: Download the AI Model File

The model file is too large for GitHub, so you need to download it separately.

1. Install the download tool:
```
pip install gdown
```

2. Download the model:
```
gdown 1ycl3dMBIMamgNC9JNmPGCMJSS9xNO8gT -O "Flask Deployed App\plant_disease_model_1_latest.pt"
```

> If `gdown` gives an error, you can manually download from this Google Drive link and place the file inside the `Flask Deployed App` folder:
> https://drive.google.com/uc?id=1ycl3dMBIMamgNC9JNmPGCMJSS9xNO8gT

> **Note:** If the downloaded file is named `plant_disease_model_1.pt`, rename it to `plant_disease_model_1_latest.pt`.

---

## Step 7: Run the App

```
cd "Flask Deployed App"
python app.py
```

You should see output like:
```
 * Running on http://127.0.0.1:5000
```

---

## Step 8: Open the App in Your Browser

Open your web browser (Chrome, Edge, etc.) and go to:

```
http://127.0.0.1:5000
```

The Plant Disease Detection app should now be running!

---

## How to Use the App

1. **Upload a Leaf Image** - Click "Choose File" or drag and drop a leaf image
2. **Use Camera** - Click the camera button to take a photo (auto-detects after capture)
3. **View Results** - The app will identify the disease and show recommended supplements
4. **Change Language** - Use the language dropdown in the navbar to translate the page
5. **Listen** - Click the speaker icon to hear the disease description read aloud
6. **Alerts** - Visit the Alerts page to see and report disease alerts in your area

---

## Stopping the App

To stop the app, go back to the Command Prompt and press **Ctrl + C**.

---

## Running Again Later

Every time you want to run the app again, open Command Prompt and type:

```
C:\plant-venv\Scripts\activate
cd "C:\plant-project\-Advanced-Deep-Neural-Network-Framework-for-Early-Plant-Disease-Identification-and-Precision-Farming\Flask Deployed App"
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python` command not found | Reinstall Python and make sure "Add to PATH" is checked |
| `pip install` fails with long path error | Make sure you created the venv at `C:\plant-venv` (short path) |
| Model file download fails | Download manually from Google Drive link above |
| App shows error on startup | Make sure you're in the `Flask Deployed App` folder |
| Camera not working | Allow camera permission when browser asks. Use Chrome for best support |
| Page won't load | Check Command Prompt for errors. Make sure the app is still running |
