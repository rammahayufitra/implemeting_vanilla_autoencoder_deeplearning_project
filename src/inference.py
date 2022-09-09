import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
from torch_snippets import *
from torchvision.datasets import MNIST  
from models import AutoEncoder
device = 'cuda:0'


# model = AutoEncoder(3).to(device)
model = torch.load('../public/models/model.pt', map_location=device)
checkpoint = torch.load('../public/weights/weights.pt', map_location=device)
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
    ret, frame = vid.read()
    frame = cv2.resize(frame, (28,28), interpolation = cv2.INTER_AREA)
    image = Image.fromarray(frame).convert('L')
    image = transform(image)
    outputs = model(image)[0][0].cpu().detach().numpy()
    print(outputs)
    cv2.imshow('input', frame)
    cv2.imshow('output', outputs)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()