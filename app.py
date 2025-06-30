import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')
import h5py
import numpy as np
import os
import cv2

from flask import Flask, request, render_template
from tensorflow import keras

# Load model
model = tf.keras.models.load_model('rice.h5', custom_objects={'KerasLayer': hub.KerasLayer})



# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Details page route
@app.route('/details')
def pred():
    return render_template('details.html')

# Result route for prediction
@app.route('/result', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'Data', 'val', f.filename)
        f.save(filepath)

        a2 = cv2.imread(filepath)
        a2 = cv2.resize(a2, (224, 224))
        a2 = np.array(a2)
        a2 = a2 / 255
        a2 = np.expand_dims(a2, 0)

        pred = model.predict(a2)
        pred = pred.argmax()

        df_labels = {
            'arborio': 0,
            'basmati': 1,
            'ipsala': 2,
            'jasmine': 3,
            'karacadag': 4
        }

        # Map prediction back to label
        prediction = ''
        for i, j in df_labels.items():
            if pred == j:
                prediction = i

        return render_template('results.html', prediction_text=prediction)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')
import h5py
import numpy as np
import os
import cv2

from flask import Flask, request, render_template
from tensorflow import keras

# Load model
model = tf.keras.models.load_model('rice.h5', custom_objects={'KerasLayer': hub.KerasLayer})

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Details page route
@app.route('/details')
def pred():
    return render_template('details.html')

# Result route for prediction
@app.route('/result', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['image']

        # Create upload directory if it doesn't exist
        upload_dir = os.path.join(os.path.dirname(__file__), 'Data', 'val')
        os.makedirs(upload_dir, exist_ok=True)

        # Save the uploaded image
        filepath = os.path.join(upload_dir, f.filename)
        f.save(filepath)

        # Preprocess the image
        a2 = cv2.imread(filepath)
        a2 = cv2.resize(a2, (224, 224))
        a2 = np.array(a2) / 255.0
        a2 = np.expand_dims(a2, axis=0)

        # Predict
        pred = model.predict(a2)
        pred_class = pred.argmax()

        # Label mapping
        df_labels = {
            0: 'arborio',
            1: 'basmati',
            2: 'ipsala',
            3: 'jasmine',
            4: 'karacadag'
        }

        prediction = df_labels.get(pred_class, "Unknown")

        return render_template('results.html', prediction_text=prediction)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
