"""
Flask app for live classification using the chosen .keras/.h5 model in `test_results`.
Rules:
- Load class order from evaluation JSON (`test_results/*_metrics_*.json`) if available.
- Otherwise derive order from a generator on `dataset/test/Fruit` (exact flow_from_directory mapping).
- Preprocess images the same way as training: resize to `IMG_SIZE` and rescale 1/255.
- Provide endpoints: `/` (UI), `/predict` (POST image), `/class_map`, `/models`.
"""

import os
from pathlib import Path
import base64
from io import BytesIO

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Paths
APP_DIR = Path(__file__).resolve().parent
RUN_DIR = APP_DIR.parent
PROJECT_ROOT = RUN_DIR.parent
TEST_RESULTS_DIR = RUN_DIR / 'test_results'
TEST_DIR = PROJECT_ROOT / 'dataset' / 'test' / 'Fruit'

# Config
IMG_SIZE = 224
DEFAULT_MODEL_NAME = 'best_model.keras'
DEFAULT_CLASS_NAMES = ['Apple', 'Banana', 'Cucumber', 'Grapefruit', 'Pomegranate']

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)


def find_model_path():
    # Prefer explicit best model file
    best = TEST_RESULTS_DIR / DEFAULT_MODEL_NAME
    if best.exists():
        return str(best)
    # Fallback: newest .keras or .h5
    if not TEST_RESULTS_DIR.exists():
        return None
    candidates = list(TEST_RESULTS_DIR.glob('*.keras')) + list(TEST_RESULTS_DIR.glob('*.h5'))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def load_class_names_from_metrics_json():
    if not TEST_RESULTS_DIR.exists():
        return None
    files = sorted(TEST_RESULTS_DIR.glob('*_metrics_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    try:
        with open(files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        cls = data.get('class_names')
        if isinstance(cls, list) and cls:
            return cls
    except Exception:
        return None
    return None


def load_class_names_from_generator():
    # Build a generator to get class_indices (exact mapping used by flow_from_directory)
    if not TEST_DIR.exists():
        return None
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    gen = datagen.flow_from_directory(
        str(TEST_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    # ordered by index
    ordered = [cls for cls, _ in sorted(gen.class_indices.items(), key=lambda x: x[1])]
    return ordered


# Load model
MODEL_PATH = find_model_path()
if MODEL_PATH is None:
    print('No model found in test_results. Place a .keras/.h5 file (e.g. best_model.keras) in that folder.')
    model = None
else:
    print('Loading model:', MODEL_PATH)
    model = load_model(MODEL_PATH)
    print('Model loaded successfully.')

# Build CLASS_NAMES in reliable order
class_names_source = None
CLASS_NAMES = None
metrics_names = load_class_names_from_metrics_json()
if metrics_names:
    CLASS_NAMES = [c.replace('_', ' ').title() for c in metrics_names]
    class_names_source = f'metrics_json ({TEST_RESULTS_DIR})'
else:
    gen_names = load_class_names_from_generator()
    if gen_names:
        CLASS_NAMES = [c.replace('_', ' ').title() for c in gen_names]
        class_names_source = 'generator.class_indices'
    else:
        raw = None
        if TEST_DIR.exists():
            raw = [p.name for p in sorted(TEST_DIR.iterdir()) if p.is_dir()]
        if raw:
            CLASS_NAMES = [c.replace('_', ' ').title() for c in raw]
            class_names_source = 'folder names'
        else:
            CLASS_NAMES = DEFAULT_CLASS_NAMES
            class_names_source = 'default'

print(f'Class names source: {class_names_source}')
print('Class names (index order):', CLASS_NAMES)


def preprocess_image(pil_img: Image.Image):
    img = pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route('/')
def index():
    return render_template('index.html', class_names=CLASS_NAMES)


@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'no_model', 'message': 'No model available on server.'}), 500

    # Accept JSON with base64 or multipart/form-data file
    data = request.get_json(silent=True)
    if data and 'image' in data:
        try:
            b64 = data['image'].split(',', 1)[-1]
            img = Image.open(BytesIO(base64.b64decode(b64)))
        except Exception as e:
            return jsonify({'error': 'bad_image', 'message': str(e)}), 400
    else:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'no_image', 'message': 'Send image as JSON (image: dataurl) or multipart file "file"'}), 400
        img = Image.open(file.stream)

    x = preprocess_image(img)
    preds = model.predict(x)
    probs = preds[0].tolist()
    idx = int(np.argmax(probs))
    conf = float(np.max(probs))

    # Map index to name using CLASS_NAMES (index order must match model)
    name = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else str(idx)

    return jsonify({
        'model_path': MODEL_PATH,
        'index': idx,
        'class_name': name,
        'confidence': conf,
        'probs': probs
    })


@app.route('/class_map', methods=['GET'])
def class_map():
    # return list of names in index order
    return jsonify({'class_names': CLASS_NAMES, 'source': class_names_source})


@app.route('/models', methods=['GET'])
def models_list():
    if not TEST_RESULTS_DIR.exists():
        return jsonify({'models': []})
    models = [p.name for p in sorted(TEST_RESULTS_DIR.glob('*.keras'))] + [p.name for p in sorted(TEST_RESULTS_DIR.glob('*.h5'))]
    return jsonify({'models': models})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
