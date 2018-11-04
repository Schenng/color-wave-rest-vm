from flask import Flask, send_file
import torch
import torchvision
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

@app.before_request
def _load_model():
    global MODEL
    MODEL = torch.load('./bucket-fuse/model')

@app.route('/')
def index():
    return "Hello, World 2!"

@app.route('/process')
def process():
    #Load an image
    A_path = 'bucket-fuse/paintschainer.jpg'
    A_img = Image.open(A_path).convert('RGB')
    ratio = A_img.width / A_img.height

    #Create transforms
    transform_list = [transforms.Resize((256,256)), transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    allTransforms = transforms.Compose(transform_list)
    
    A_img = torch.stack([A_img] * 3, dim = 0).unsqueeze(0)

    return send_file(A_img, mimetype='image/gif')

@app.route('/fuse')
def fuse():
    filename = 'bucket-fuse/paintschainer.jpg'
    return send_file(filename, mimetype='image/gif')

@app.route('/image')
def image():
    return "Image Response"

if __name__ == '__main__':
    #app.run(debug=True)
    app.run()
