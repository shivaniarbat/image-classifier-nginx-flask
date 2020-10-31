import json
import os

import re
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from flask import Flask, request, jsonify
#from pymongo import MongoClient

app = Flask(__name__)

PATH = "/project/densenet121-a639ec97.pth"
#model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
#model = models.densenet121()

# read all lines at once
#state_dict_data = torch.load(PATH)

#model.eval()

def _load_state_dict(model):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = torch.load(PATH)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)

def _densenet(arch, growth_rate, block_config, num_init_features, pretrained ):
    model = models.DenseNet(growth_rate, block_config, num_init_features)
    if pretrained:
        _load_state_dict(model)
    return model

pretrained = True
model = _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained)

model.eval()

##
img_class_map = None
mapping_file_path = '/project/index_to_name.json'                  # Human-readable names for Imagenet classes
if os.path.isfile(mapping_file_path):
    with open (mapping_file_path) as f:
        img_class_map = json.load(f)



# Transform input into the form our model expects
def transform_image(infile):
    print('inside transform fn updated ')
    input_transforms = [transforms.Resize(255),           # We use multiple TorchVision transforms to ready the image
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],       # Standard normalization for ImageNet model input
            [0.229, 0.224, 0.225])]
    print("input_transforms:",input_transforms)
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)                            # Open the image file
    timg = my_transforms(image)                           # Transform PIL image to appropriately-shaped PyTorch tensor
    timg.unsqueeze_(0)                                    # PyTorch models expect batched input; create a batch of 1
    print("image transform output:",timg)
    return timg

# Get a prediction
def get_prediction(input_tensor):
    print('')
    outputs = model.forward(input_tensor)                 # Get likelihoods for all ImageNet classes
    _, y_hat = outputs.max(1)                             # Extract the most likely class
    prediction = y_hat.item()                             # Extract the int value from the PyTorch tensor
    return prediction

# Make the prediction human-readable
def render_prediction(prediction_idx):
    stridx = str(prediction_idx)
    class_name = 'Unknown'
    if img_class_map is not None:
        if stridx in img_class_map is not None:
            class_name = img_class_map[stridx][1]

    return prediction_idx, class_name


@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict-image-label endpoint with an RGB image attachment'})


@app.route('/predict-image-label', methods=['POST','GET'])
def predict():
    print('received the POST request new')
    if request.method == 'POST':
        print('before file')
        file = request.files['file']
        print('after file')
        if file is not None:
            print('before tensor')
            input_tensor = transform_image(file)
            print("input-tensor created")
            prediction_idx = get_prediction(input_tensor)
            print("prediction done")
            class_id, class_name = render_prediction(prediction_idx)
            print("class id:",class_id)
            #return jsonify({'class_id': class_id, 'class_name': class_name})
            return 'successful'
        else:
            return 'something went wrong'
    else:
        print("nothing was executed. request is not POST")
    return 'unsuccesful'

@app.route("/ping")
def ping():
    return "pong"

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
