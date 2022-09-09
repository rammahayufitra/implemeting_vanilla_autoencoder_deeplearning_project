import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
from torch_snippets import *
from torchvision.datasets import MNIST  
from models import AutoEncoder
device = 'cuda:0'


model = AutoEncoder(3).to(device)
model = torch.load('../public/models/model.pt', map_location=device)
checkpoint = torch.load('../public/weights/weight.pt', map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize([0.5], [0.5]),
    transforms.Lambda(lambda x: x.to(device))
])

vid = cv2.VideoCapture("https://192.168.1.12:8080/video")

value = []
while(True):
    ret, image = vid.read()
    image = cv2.resize(image, (32,32), interpolation = cv2.INTER_AREA)
    # image = Image.fromarray(image).convert('L')
    image = transform(image)
    # image = image.unsqueeze(0).to(device)
    outputs = model(image)[0]
    print(outputs)
    # label = np.array(outputs.detach()).argmax()   
    # value.append(label)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # org = (00, 185)
    # fontScale = 1
    # color = (0, 0, 255)
    # thickness = 2
    # image = cv2.putText(image, str(label), org, font, fontScale, 
    #                 color, thickness, cv2.LINE_AA, False)
    # cv2.imshow('frame1', frame1)
    # cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()