import flask
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pickle
import numpy as np
from util import base64_to_pil
import torch
import cv2
from torchvision import models
import torch.nn as nn

# Running the flask app
app = Flask(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load model using pickle
# model = pickle.load(open('model.pkl', 'rb'))

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("./best_model_resnet18.pt", map_location=DEVICE))

model.eval()
model.to(DEVICE)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# @app.route('/upload')
# def upload_file():
#    return render_template('upload.html')

# @app.route('/uploader', methods = ['GET', 'POST'])
# def upload_file():
#    if request.method == 'POST':
#       f = request.files['file']
#       f.save(secure_filename(f.filename))
#       return 'file uploaded successfully'


def model_predict(image, model):
    '''
    Prediction Function for model.
    Arguments: 
        image: is address to image
        model : image classification model
    '''

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (224, 224))

    image = np.transpose(image, (2, 0, 1)).astype(np.float32) # (c, h, w)

    image = (torch.Tensor(image) / 255.0) # (c, h, w)

    pred = model(image.unsqueeze(0)).argmax(dim=1)
    return pred

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    predict function to predict the image
    Api hits this function when someone clicks submit.
    '''
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Make prediction
        pred = model_predict(img, model).detach().cpu().item()
        if (pred == 0):
            result = "undamaged"
        else:
            result = "damaged"
        
        return jsonify(result=result)
    return None

if __name__ == '__main__':
    app.run(debug=True)
