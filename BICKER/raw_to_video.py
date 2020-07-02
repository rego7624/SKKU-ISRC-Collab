import matplotlib
matplotlib.use('TkAgg')

import torch
import torchvision.transforms as transforms
import cv2
from matplotlib import pyplot as plt
import os
from PIL import Image
from run import save_image

import network
import config
import dataset
from extract_from_img import extractFromImg2, plot_lm

from webcam_demo.webcam_extraction_conversion import *

cpu = torch.device("cpu")


#Embedder Test --> Okay, Landmark Test --> Okay
#y_t = torch.load('y_t.tar', map_location=cpu)
#y_t = y_t['y_t']
#e_hat1 = e_hat1[0].unsqueeze(0)

print('PRESS Q TO EXIT')
#plt.ion()

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')

trans = transforms.ToTensor()


cap = cv2.VideoCapture('outpyjj.avi')
out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (256,256)) #

with torch.no_grad():
    while(cap.isOpened):
        _, x = cap.read()
        #x = x[100:620, 380:900, :]
        #x_im = Image.fromarray(x)
        #x = x_im.resize((256,256))
        x,_,x_lm = extractFromImg2(x)
        #        x = x[140:620, 400:840, :]
        #        x_im = Image.fromarray(x)
        #        x = x_im.resize((256,256))
        #x = np.transpose(x, (1,0,2))
        
        
        #x = trans(x).unsqueeze(0)
        
        #g_y = trans(g_y).unsqueeze(0)
        
        #x_hat = G(g_y, e_hat)
        #x_hat = (x_hat[0].transpose(0,2).transpose(0,1).to(cpu).numpy()*255).astype(np.uint8)

        #x_hat = cv2.cvtColor(x_hat, cv2.COLOR_BGR2RGB)
        
        #        g_y = g_y.unsqueeze(0)
        #        x = x.unsqueeze(0)
        #plt.clf()
        #       save_image(f'{datetime.now():%Y%m%d_%H%M%S%f}_x_hat.png',x_hat[0])
        #save_image(f'{datetime.now():%Y%m%d_%H%M%S%f}_x.png',x[0])
        #print(x_hat.shape)
        #out1 = x_hat[0].transpose(0,2).transpose(0,1)
        #out1 = out1.to(cpu).numpy()
        print("hello")
        out.write(x_hat)
#
#        out2 = x[0].transpose(0,2).transpose(0,1)
#        out2 = out2.to(cpu).numpy()
#
#        out3 = g_y[0].transpose(0,2).transpose(0,1)
#        out3 = out3.to(cpu).numpy()
#
#        cv2.imshow('me', out2)
#        cv2.moveWindow('me',0,0)
#        cv2.imshow('ladnmark', out3)
#        cv2.moveWindow('landmark',400,0)
#        cv2.imshow('fake', cv2.cvtColor(out1, cv2.COLOR_BGR2RGB))
#        cv2.moveWindow('fake',800,0)
        #plt.show()
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()



