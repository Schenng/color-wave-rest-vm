from flask import Flask
import torch
import torchvision
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World 2!"

@app.route('/process')
def process():
    
    return 'Process'

@app.route('/image')
def image():
    return "Image Response"

if __name__ == '__main__':
    #app.run(debug=True)
    app.run()
