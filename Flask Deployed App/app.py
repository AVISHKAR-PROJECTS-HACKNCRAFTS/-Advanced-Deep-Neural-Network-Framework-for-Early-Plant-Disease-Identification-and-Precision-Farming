import os
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from deep_translator import GoogleTranslator
from supabase_client import get_supabase


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location=torch.device('cpu'), weights_only=True))
model.eval()

# ── Constants ──────────────────────────────────────────────────────────────

HEALTHY_INDICES = {3, 5, 7, 11, 15, 18, 20, 23, 24, 25, 28, 38}

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

# ── Translation ────────────────────────────────────────────────────────────

translation_cache = {}

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


@app.route('/')
@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files.get('image')
        if not image or not image.filename:
            return redirect('/')
        filename = image.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(file_path)

        result = prediction(file_path)
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

        return render_template('submit.html',
            title=title, desc=description, prevent=prevent,
            image_url=image_url, pred=pred,
            sname=supplement_name, simage=supplement_image_url,
            buy_link=supplement_buy_link,
            confidence=confidence, severity=severity,
            confidence_color=confidence_color,
            alternatives=alternatives, is_healthy=is_healthy)

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

@app.route('/api/translate')
def translate_text():
    text = request.args.get('text', '')
    dest = request.args.get('dest', 'en')

    if not text or dest == 'en':
        return jsonify({'translated': text})

    cache_key = (hash(text[:200]), dest)
    if cache_key in translation_cache:
        return jsonify({'translated': translation_cache[cache_key]})

    try:
        translated = GoogleTranslator(source='en', target=dest).translate(text)
        translation_cache[cache_key] = translated
        return jsonify({'translated': translated})
    except Exception as e:
        print(f"Translation error: {e}")
        return jsonify({'translated': text, 'error': str(e)})

# ── Community Alerts ───────────────────────────────────────────────────────

@app.route('/alerts')
def alerts_page():
    return render_template('alerts.html')

@app.route('/api/alerts')
def get_alerts():
    sb = get_supabase()
    if not sb:
        return jsonify({'error': 'Database not configured', 'alerts': []}), 503

    region = request.args.get('region', '')

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

@app.route('/api/report-alert', methods=['POST'])
def report_alert():
    sb = get_supabase()
    if not sb:
        return jsonify({'error': 'Database not configured'}), 503

    data = request.get_json()

    required_fields = ['disease_name', 'disease_index', 'severity', 'confidence']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400

    alert = {
        'disease_name': data['disease_name'],
        'disease_index': int(data['disease_index']),
        'severity': data['severity'],
        'confidence': float(data['confidence']),
        'latitude': data.get('latitude'),
        'longitude': data.get('longitude'),
        'region_name': data.get('region_name', 'Unknown'),
        'image_url': data.get('image_url', ''),
        'description': data.get('description', ''),
        'reported_by': data.get('reported_by', 'anonymous'),
    }

    try:
        response = sb.table('disease_alerts').insert(alert).execute()
        return jsonify({'success': True, 'alert': response.data[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/subscribe', methods=['POST'])
def subscribe_alerts():
    sb = get_supabase()
    if not sb:
        return jsonify({'error': 'Database not configured'}), 503

    data = request.get_json()
    email = data.get('email', '').strip()
    region = data.get('region_name', '').strip()

    if not email or not region:
        return jsonify({'error': 'Email and region are required'}), 400

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


if __name__ == '__main__':
    app.run(debug=True)
