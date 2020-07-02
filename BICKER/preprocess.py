import matplotlib
matplotlib.use('TkAgg')

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2
from matplotlib import pyplot as plt
import os
from PIL import Image
from run import save_image



import network
import config
import dataset
from extract_from_img import extractFromImg, plot_lm, lm_adapt

from webcam_demo.webcam_extraction_conversion import *

cpu = torch.device("cpu")

print('PRESS Q TO EXIT')

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')
tar, tar_lm1, tarlm = extractFromImg("targets/Oabama.jpg")

trans = transforms.ToTensor()

tar = trans(tar).unsqueeze(0)
tar_lm = trans(tar_lm1).unsqueeze(0)

def preProcess(img, tar_lm):
        EyeCenter = tar_lm[36:48,1].mean()
        Chin = tar_lm[8,1]
        center = tar_lm[0:17].mean()
        
        face_ratio = 110.0 / (Chin - EyeCenter)
        
        size = face_ratio * 256
        
        transform=transforms.Compose([
            transforms.Resize(size)
        ])
        
        img = transform(img)
        
        img = TF.resized_crop(img, center-128, center-128, 256, 256, size=256)
        return img

