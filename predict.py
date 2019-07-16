import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
from torch.optim import lr_scheduler
import copy
import json
import os
from os.path import exists
import argparse

train_on_gpu = torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location="cpu")
    model = models.resnet152()
    
    input_size = 2048
    output_size = 2
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 512)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(512, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.fc = classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint['class_to_idx']

loaded_model, class_to_idx = load_checkpoint('8960_checkpoint.pth')
idx_to_class = { v : k for k,v in class_to_idx.items()}

def process_image(image):
    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npImage = np.array(image)
    npImage = npImage/255.
        
    imgA = npImage[:,:,0]
    imgB = npImage[:,:,1]
    imgC = npImage[:,:,2]
    
    imgA = (imgA - 0.485)/(0.229) 
    imgB = (imgB - 0.456)/(0.224)
    imgC = (imgC - 0.406)/(0.225)
        
    npImage[:,:,0] = imgA
    npImage[:,:,1] = imgB
    npImage[:,:,2] = imgC
    
    npImage = np.transpose(npImage, (2,0,1))
    
    return npImage

def predict(image_path, model, topk=2):
    
    image = torch.FloatTensor([process_image(Image.open(image_path))])
    model.eval()
    output = model.forward(Variable(image))
    pobabilities = torch.exp(output).data.numpy()[0]
    

    top_idx = np.argsort(pobabilities)[-topk:][::-1] 
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = pobabilities[top_idx]

    return top_probability, top_class


parser = argparse.ArgumentParser(description='Determine if breast cancer cells are malignant using ML.')
parser.add_argument('-f','--files', action='append', help='<Required> Set flag', required=True)


args = parser.parse_args()
results = []

for file in args.files:
  result = predict(file, loaded_model)
  results.append({
      "confidence": result[0][0],
      "classification": result[1][0]
    })
print(results)

# def run_all_tests():
#   malignant_correct = 0
#   malignant_total = 0
#   for filename in os.listdir(path):
#     malignant_total = malignant_total + 1
#     print("Image " + str(malignant_total) + "/890: ")
#     result = predict(path + filename, loaded_model)
#     if(((result[0][0] > 0.5) and (result[1][0] == "MALIGNANT")) or ((result[0][1] > 0.5) and (result[1][1] == "MALIGNANT")) ): 
#       malignant_correct = malignant_correct + 1
#       print("Correctly identified as malignant!")
#     else: 
#       print("Incorrectly identified as benign!")
#     print("Confidence: " + str(result[0][0]))
#     print("Correct: " + str(malignant_correct) + "/" + str(malignant_total))
#     print("\n\n")

# run_all_tests()
