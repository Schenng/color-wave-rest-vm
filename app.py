from flask import Flask, send_file, request
import torch
import torchvision
import numpy as np
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
from scipy.misc import imresize


app = Flask(__name__)

@app.before_request
def _load_model():
    global MODEL
    MODEL = torch.load('./bucket-fuse/model')

@app.route('/')
def index():
    return "Color Wave Backend"

# Process()
#
# Takes a filename and looks up the file within Google fuse, then processes the image through the model,
# then returns the processed image
#
# Params:
# String filename - used to reference the file on Google Fuse
#
@app.route('/process', methods=['POST'])
def image():

    filename = request.get_json().get('filename')
    image_path = ('bucket-fuse/' + filename)
    A_img = Image.open(image_path).convert('RGB')

    aspect_ratio = float(A_img.width) / float(A_img.height)

    #Create transforms
    transform_list = [transforms.Resize((256,256)), transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    allTransforms = transforms.Compose(transform_list)

    # Do transforms
    A_img = allTransforms(A_img)
    A_img = A_img[0, ...] * 0.299 + A_img[1, ...] * 0.587 + A_img[2, ...] * 0.114

    #Some more transforms?
    A_img = torch.stack([A_img] * 3, dim = 0).unsqueeze(0)

    # Put the image through the model
    MODEL.set_input({'A': A_img, 'A_paths' : ''})
    MODEL.test()    
    visuals = MODEL.get_current_visuals()

    im_data = visuals['fake_B']

    im = tensor2im(im_data)
    image_name = '%s_processed.png' % (filename)
    h, w, _ = im.shape
    if aspect_ratio > 1.0:
        im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
    if aspect_ratio < 1.0:
        im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')

    image_pil = Image.fromarray(im)

    img_io = BytesIO()
    image_pil.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/gif') 

@app.route('/testprocess')
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

    # Do transforms
    A_img = allTransforms(A_img)
    A_img = A_img[0, ...] * 0.299 + A_img[1, ...] * 0.587 + A_img[2, ...] * 0.114

    #Some more transforms?
    A_img = torch.stack([A_img] * 3, dim = 0).unsqueeze(0)

    # Put the image through the model
    MODEL.set_input({'A': A_img, 'A_paths' : ''})
    MODEL.test()    
    visuals = MODEL.get_current_visuals()

    return save_images(visuals, 'bucket-fuse/', aspect_ratio=1.0, width=256)

@app.route('/testimage')
def fuse():
    filename = 'bucket-fuse/paintschainer.jpg'
    return send_file(filename, mimetype='image/gif')

if __name__ == '__main__':
    #app.run(debug=True)
    app.run()

def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def save_images(visuals, image_path, aspect_ratio=1.0, width=256):
    flag = 0

    for label, im_data in visuals.items():
        im = tensor2im(im_data)
        image_name = '%s.png' % (label)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')

        image_pil = Image.fromarray(im)

        img_io = BytesIO()
        image_pil.save(img_io, 'JPEG', quality=70)
        img_io.seek(0)
        if(flag == 0):
            flag = 1
        else: 
            return send_file(img_io, mimetype='image/gif') 
        #image_pil.save(image_path + image_name)