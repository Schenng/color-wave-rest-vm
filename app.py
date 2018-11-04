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

    #Load an image
    A_path = './images/img_0001.jpg'
    A_img = Image.open(A_path).convert('RGB')
    ratio = A_img.width / A_img.height

    #Create transforms
    transform_list = [transforms.Resize((256,256)), transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    allTransforms = transforms.Compose(transform_list)

    return 'process image'
@app.route('/image')
def image():
    return "Image Response"

if __name__ == '__main__':
    #app.run(debug=True)
    app.run()
