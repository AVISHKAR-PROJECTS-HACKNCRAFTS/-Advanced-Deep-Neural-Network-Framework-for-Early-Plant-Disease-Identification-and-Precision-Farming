import os
import sqlite3
import time
import logging
from flask import Flask, redirect, render_template, request, jsonify, make_response
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from deep_translator import GoogleTranslator
from supabase_client import get_supabase

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ── CSV Loading ───────────────────────────────────────────────────────────
try:
    disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
    supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')
    logger.info("CSV data files loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"CSV data file not found: {e}. The app will not function correctly.")
    disease_info = pd.DataFrame()
    supplement_info = pd.DataFrame()
except Exception as e:
    logger.error(f"Error loading CSV data files: {e}. The app will not function correctly.")
    disease_info = pd.DataFrame()
    supplement_info = pd.DataFrame()

# ── Model Loading ─────────────────────────────────────────────────────────
model = CNN.CNN(39)
try:
    model.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    logger.info("Model loaded successfully.")
except FileNotFoundError:
    logger.error("Model file 'plant_disease_model_1_latest.pt' not found. Predictions will be unavailable.")
    model = None
except Exception as e:
    logger.error(f"Error loading model: {e}. Predictions will be unavailable.")
    model = None

# ── Constants ──────────────────────────────────────────────────────────────

HEALTHY_INDICES = {3, 5, 7, 11, 15, 18, 20, 23, 24, 25, 28, 38}

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

SEVERITY_MAP = {
    0: 'Moderate',    # Apple Scab
    1: 'Critical',    # Apple Black Rot
    2: 'Moderate',    # Apple Cedar Rust
    3: 'Healthy',     # Apple Healthy
    4: 'N/A',         # Background
    5: 'Healthy',     # Blueberry Healthy
    6: 'Moderate',    # Cherry Powdery Mildew
    7: 'Healthy',     # Cherry Healthy
    8: 'Critical',    # Corn Cercospora / Gray Leaf Spot
    9: 'Moderate',    # Corn Common Rust
    10: 'Critical',   # Corn Northern Leaf Blight
    11: 'Healthy',    # Corn Healthy
    12: 'Critical',   # Grape Black Rot
    13: 'Critical',   # Grape Esca (Black Measles)
    14: 'Moderate',   # Grape Leaf Blight
    15: 'Healthy',    # Grape Healthy
    16: 'Critical',   # Orange Huanglongbing
    17: 'Moderate',   # Peach Bacterial Spot
    18: 'Healthy',    # Peach Healthy
    19: 'Moderate',   # Pepper Bacterial Spot
    20: 'Healthy',    # Pepper Healthy
    21: 'Moderate',   # Potato Early Blight
    22: 'Critical',   # Potato Late Blight
    23: 'Healthy',    # Potato Healthy
    24: 'Healthy',    # Raspberry Healthy
    25: 'Healthy',    # Soybean Healthy
    26: 'Mild',       # Squash Powdery Mildew
    27: 'Moderate',   # Strawberry Leaf Scorch
    28: 'Healthy',    # Strawberry Healthy
    29: 'Moderate',   # Tomato Bacterial Spot
    30: 'Moderate',   # Tomato Early Blight
    31: 'Critical',   # Tomato Late Blight
    32: 'Mild',       # Tomato Leaf Mold
    33: 'Moderate',   # Tomato Septoria Leaf Spot
    34: 'Mild',       # Tomato Spider Mites
    35: 'Moderate',   # Tomato Target Spot
    36: 'Critical',   # Tomato Yellow Leaf Curl Virus
    37: 'Critical',   # Tomato Mosaic Virus
    38: 'Healthy',    # Tomato Healthy
}

# ── Rate Limiting ─────────────────────────────────────────────────────────

# Simple in-memory rate limiter: { ip_string: [timestamp, timestamp, ...] }
_rate_limit_store = {}
RATE_LIMIT_MAX_REQUESTS = 30
RATE_LIMIT_WINDOW_SECONDS = 60


def _is_rate_limited(ip):
    """Return True if ip has exceeded RATE_LIMIT_MAX_REQUESTS in the last window."""
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS

    if ip not in _rate_limit_store:
        _rate_limit_store[ip] = []

    # Prune old entries
    _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if t > cutoff]

    if len(_rate_limit_store[ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return True

    _rate_limit_store[ip].append(now)
    return False


# ── Translation ────────────────────────────────────────────────────────────

translation_cache = {}

# ── Helper: file extension check ──────────────────────────────────────────

def _allowed_file(filename):
    """Return True if the filename has an allowed image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Prediction ─────────────────────────────────────────────────────────────

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)

    probabilities = F.softmax(output, dim=1)
    probabilities = probabilities.detach().numpy().flatten()

    top3_indices = np.argsort(probabilities)[::-1][:3]
    top3_probs = probabilities[top3_indices]

    primary_index = int(top3_indices[0])
    primary_confidence = float(top3_probs[0])
    severity = SEVERITY_MAP.get(primary_index, 'Unknown')

    alternatives = []
    for i in range(1, len(top3_indices)):
        alt_idx = int(top3_indices[i])
        alt_prob = float(top3_probs[i])
        alt_name = disease_info['disease_name'][alt_idx]
        alternatives.append({
            'index': alt_idx,
            'name': alt_name,
            'confidence': round(alt_prob * 100, 1),
            'severity': SEVERITY_MAP.get(alt_idx, 'Unknown')
        })

    return {
        'index': primary_index,
        'confidence': round(primary_confidence * 100, 1),
        'severity': severity,
        'alternatives': alternatives
    }


# ── Flask App ──────────────────────────────────────────────────────────────

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── SQLite Fallback for Alerts ────────────────────────────────────────────
ALERTS_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alerts.db')

def init_sqlite():
    conn = sqlite3.connect(ALERTS_DB)
    conn.execute('''CREATE TABLE IF NOT EXISTS disease_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        disease_name TEXT, disease_index INTEGER, severity TEXT,
        confidence REAL, latitude REAL, longitude REAL,
        region_name TEXT, image_url TEXT, description TEXT,
        reported_by TEXT DEFAULT 'anonymous',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS alert_subscriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT, region_name TEXT,
        UNIQUE(email, region_name)
    )''')
    conn.commit()
    conn.close()

init_sqlite()

def get_sqlite():
    conn = sqlite3.connect(ALERTS_DB)
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/')
@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method != 'POST':
        return redirect('/')

    # Check if model is available
    if model is None:
        logger.error("Prediction requested but model is not loaded.")
        return render_template('index.html', error="Model is not available. Please contact the administrator."), 503

    image = request.files.get('image')
    if not image or not image.filename:
        return render_template('index.html', error="Please select an image to upload.")

    filename = image.filename

    # Validate file extension
    if not _allowed_file(filename):
        logger.warning(f"Upload rejected: disallowed extension for '{filename}'")
        return render_template('index.html', error="Invalid file type. Please upload a JPG, PNG, or WEBP image.")

    # Validate file size (read content length; seek back for saving)
    image.seek(0, os.SEEK_END)
    file_size = image.tell()
    image.seek(0)
    if file_size > MAX_FILE_SIZE:
        logger.warning(f"Upload rejected: file too large ({file_size} bytes)")
        return render_template('index.html', error="File is too large. Maximum size is 10MB.")

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(file_path)

    # Validate that the file is a real image (not corrupt)
    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception as e:
        logger.warning(f"Upload rejected: corrupt or invalid image file '{filename}': {e}")
        try:
            os.remove(file_path)
        except OSError:
            pass
        return render_template('index.html', error="The uploaded file appears to be corrupt. Please try another image.")

    try:
        result = prediction(file_path)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        try:
            os.remove(file_path)
        except OSError:
            pass
        return render_template('index.html', error="An error occurred during analysis. Please try again.")

    # Keep uploaded file for display on results page
    # (will be overwritten on next upload with same name)
    uploaded_image_url = f"/static/uploads/{filename}"

    pred = result['index']
    confidence = result['confidence']
    severity = result['severity']
    alternatives = result['alternatives']

    title = disease_info['disease_name'][pred]
    description = disease_info['description'][pred]
    prevent = disease_info['Possible Steps'][pred]
    image_url = disease_info['image_url'][pred]
    supplement_name = supplement_info['supplement name'][pred]
    supplement_image_url = supplement_info['supplement image'][pred]
    supplement_buy_link = supplement_info['buy link'][pred]

    if confidence >= 90:
        confidence_color = 'high'
    elif confidence >= 70:
        confidence_color = 'medium'
    else:
        confidence_color = 'low'

    is_healthy = pred in HEALTHY_INDICES

    resp = make_response(render_template('submit.html',
        title=title, desc=description, prevent=prevent,
        image_url=image_url, pred=pred,
        uploaded_image_url=uploaded_image_url,
        sname=supplement_name, simage=supplement_image_url,
        buy_link=supplement_buy_link,
        confidence=confidence, severity=severity,
        confidence_color=confidence_color,
        alternatives=alternatives, is_healthy=is_healthy))

    # Add no-cache headers for prediction results
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link']),
        healthy_indices=HEALTHY_INDICES)

# ── Voice Search API ──────────────────────────────────────────────────────

@app.route('/api/search-disease')
def search_disease():
    q = request.args.get('q', '').lower().strip()
    if not q:
        return jsonify({'results': []})

    # Max length check on query
    if len(q) > 200:
        q = q[:200]

    results = []
    for i in range(len(disease_info)):
        name = str(disease_info['disease_name'][i]).lower()
        if q in name or any(word in name for word in q.split()):
            is_healthy = i in HEALTHY_INDICES
            results.append({
                'index': i,
                'disease_name': disease_info['disease_name'][i],
                'supplement_name': supplement_info['supplement name'][i],
                'supplement_image': supplement_info['supplement image'][i],
                'buy_link': supplement_info['buy link'][i],
                'severity': SEVERITY_MAP.get(i, 'Unknown'),
                'is_healthy': is_healthy,
                'description': disease_info['description'][i][:200],
            })

    return jsonify({'results': results})

# ── Translation API ────────────────────────────────────────────────────────

@app.route('/api/translate', methods=['GET', 'POST'])
def translate_text():
    # Rate limiting
    client_ip = request.remote_addr or 'unknown'
    if _is_rate_limited(client_ip):
        logger.warning(f"Rate limit exceeded for /api/translate from IP {client_ip}")
        return jsonify({'error': 'Rate limit exceeded. Try again later.'}), 429

    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        text = data.get('text', '')
        dest = data.get('dest', 'en')
    else:
        text = request.args.get('text', '')
        dest = request.args.get('dest', 'en')

    if not text or dest == 'en':
        return jsonify({'translated': text})

    cache_key = (hash(text), dest)
    if cache_key in translation_cache:
        return jsonify({'translated': translation_cache[cache_key]})

    try:
        translated = GoogleTranslator(source='en', target=dest).translate(text)
        translation_cache[cache_key] = translated
        return jsonify({'translated': translated})
    except Exception as e:
        logger.error(f"Translation error for text '{text[:80]}...' to '{dest}': {e}")
        return jsonify({'translated': text, 'error': str(e)})

@app.route('/api/translate-batch', methods=['POST'])
def translate_batch():
    """Translate multiple texts in a single request to avoid rate limiting."""
    import time as _time
    data = request.get_json(silent=True) or {}
    texts = data.get('texts', [])
    dest = data.get('dest', 'en')

    if not texts or dest == 'en':
        return jsonify({'translations': texts})

    results = []
    translator = GoogleTranslator(source='en', target=dest)
    for text in texts:
        if not text or not text.strip():
            results.append(text)
            continue

        cache_key = (hash(text), dest)
        if cache_key in translation_cache:
            results.append(translation_cache[cache_key])
            continue

        try:
            # Truncate very long texts to avoid Google Translate limits
            t = text[:5000] if len(text) > 5000 else text
            translated = translator.translate(t)
            translation_cache[cache_key] = translated
            results.append(translated)
            _time.sleep(0.1)  # small delay between requests
        except Exception as e:
            logger.error(f"Batch translation error for text '{text[:80]}...' to '{dest}': {e}")
            results.append(text)
            _time.sleep(0.5)

    return jsonify({'translations': results})

# ── Community Alerts ───────────────────────────────────────────────────────

@app.route('/alerts')
def alerts_page():
    return render_template('alerts.html')

@app.route('/api/alerts')
def get_alerts():
    region = request.args.get('region', '')

    sb = get_supabase()
    if sb:
        try:
            query = sb.table('disease_alerts') \
                .select('*') \
                .order('created_at', desc=True) \
                .limit(100)
            if region:
                query = query.ilike('region_name', f'%{region}%')
            response = query.execute()
            return jsonify({'alerts': response.data})
        except Exception as e:
            return jsonify({'error': str(e), 'alerts': []}), 500

    # SQLite fallback
    try:
        conn = get_sqlite()
        if region:
            rows = conn.execute(
                'SELECT * FROM disease_alerts WHERE region_name LIKE ? ORDER BY created_at DESC LIMIT 100',
                (f'%{region}%',)
            ).fetchall()
        else:
            rows = conn.execute(
                'SELECT * FROM disease_alerts ORDER BY created_at DESC LIMIT 100'
            ).fetchall()
        conn.close()
        return jsonify({'alerts': [dict(row) for row in rows]})
    except Exception as e:
        return jsonify({'error': str(e), 'alerts': []}), 500

@app.route('/api/report-alert', methods=['POST'])
def report_alert():
    data = request.get_json(silent=True) or {}

    required_fields = ['disease_name', 'disease_index', 'severity', 'confidence']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400

    # Validate disease_index
    try:
        disease_index = int(data['disease_index'])
    except (ValueError, TypeError):
        return jsonify({'error': 'disease_index must be an integer'}), 400
    if disease_index < 0 or disease_index > 38:
        return jsonify({'error': 'disease_index must be between 0 and 38'}), 400

    # Validate confidence
    try:
        confidence = float(data['confidence'])
    except (ValueError, TypeError):
        return jsonify({'error': 'confidence must be a number'}), 400
    if confidence < 0 or confidence > 100:
        return jsonify({'error': 'confidence must be between 0 and 100'}), 400

    # Validate latitude if provided
    latitude = data.get('latitude')
    if latitude is not None:
        try:
            latitude = float(latitude)
        except (ValueError, TypeError):
            return jsonify({'error': 'latitude must be a number'}), 400
        if latitude < -90 or latitude > 90:
            return jsonify({'error': 'latitude must be between -90 and 90'}), 400

    # Validate longitude if provided
    longitude = data.get('longitude')
    if longitude is not None:
        try:
            longitude = float(longitude)
        except (ValueError, TypeError):
            return jsonify({'error': 'longitude must be a number'}), 400
        if longitude < -180 or longitude > 180:
            return jsonify({'error': 'longitude must be between -180 and 180'}), 400

    alert = {
        'disease_name': data['disease_name'],
        'disease_index': disease_index,
        'severity': data['severity'],
        'confidence': confidence,
        'latitude': latitude,
        'longitude': longitude,
        'region_name': data.get('region_name', 'Unknown'),
        'image_url': data.get('image_url', ''),
        'description': data.get('description', ''),
        'reported_by': data.get('reported_by', 'anonymous'),
    }

    sb = get_supabase()
    if sb:
        try:
            response = sb.table('disease_alerts').insert(alert).execute()
            return jsonify({'success': True, 'alert': response.data[0]})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # SQLite fallback
    try:
        conn = get_sqlite()
        cursor = conn.execute(
            '''INSERT INTO disease_alerts
               (disease_name, disease_index, severity, confidence,
                latitude, longitude, region_name, image_url, description, reported_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (alert['disease_name'], alert['disease_index'], alert['severity'],
             alert['confidence'], alert['latitude'], alert['longitude'],
             alert['region_name'], alert['image_url'], alert['description'],
             alert['reported_by'])
        )
        conn.commit()
        alert['id'] = cursor.lastrowid
        alert['created_at'] = None
        conn.close()
        return jsonify({'success': True, 'alert': alert})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/subscribe', methods=['POST'])
def subscribe_alerts():
    data = request.get_json(silent=True) or {}
    email = data.get('email', '').strip()
    region = data.get('region_name', '').strip()

    if not email or not region:
        return jsonify({'error': 'Email and region are required'}), 400

    sb = get_supabase()
    if sb:
        try:
            response = sb.table('alert_subscriptions').insert({
                'email': email,
                'region_name': region
            }).execute()
            return jsonify({'success': True})
        except Exception as e:
            if 'duplicate' in str(e).lower() or '23505' in str(e):
                return jsonify({'success': True, 'message': 'Already subscribed'})
            return jsonify({'error': str(e)}), 500

    # SQLite fallback
    try:
        conn = get_sqlite()
        conn.execute(
            'INSERT INTO alert_subscriptions (email, region_name) VALUES (?, ?)',
            (email, region)
        )
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except sqlite3.IntegrityError:
        return jsonify({'success': True, 'message': 'Already subscribed'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
