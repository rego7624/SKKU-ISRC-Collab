import PIL
from PIL import Image
import face_alignment
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
#from preprocess import preProcess

def extractFromImg2(tar2):

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')

    tar = np.array(tar2)
    #tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

    tar_lm = fa.get_landmarks(tar)[0]
    print("1")
    print(tar2.shape)
    print(type(tar2))
    tar = preProcess2(tar2, tar_lm)
    print("11")
    print(tar.shape)
    print(type(tar))
    tar = np.array(tar)
    #tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)
    tar_lm = fa.get_landmarks(tar)[0]
    tar_lm = tar_lm*256/tar.shape[0]
    tar_lm = tar_lm.astype('int32')
    #Make it tensor, add dimension
    #tar = trans(target)
    #ar_lm = trans(target_lm)

    return tar, plot_lm(tar, tar_lm), tar_lm

def extractFromImg(imgAdd):
    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')
    
    tar2 = Image.open(imgAdd)
    tar = np.array(tar2)
    #tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)
    
    tar_lm = fa.get_landmarks(tar)[0]
    if True :
        tar = preProcess(tar2, tar_lm)
    tar = np.array(tar)
    #tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)
    tar_lm = fa.get_landmarks(tar)[0]
    #Make it tensor, add dimension
    #tar = trans(target)
    #ar_lm = trans(target_lm)
    
    return tar, plot_lm(tar, tar_lm), tar_lm

def plot_lm(target, landmarks):
    
    dpi = 100
    fig = plt.figure(figsize=(256/dpi, 256/dpi), dpi = dpi)
    ax = fig.add_subplot(111)
    ax.axis('off')
    print(target.shape)
    ax.imshow(np.ones([256,256,3]))# target.shape))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Head
    ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], linestyle='-', color='green', lw=2)
    # Eyebrows
    ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], linestyle='-', color='orange', lw=2)
    ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], linestyle='-', color='orange', lw=2)
    # Nose
    ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], linestyle='-', color='blue', lw=2)
    
    ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], linestyle='-', color='blue', lw=2)
    # Eyes
    ax.plot(landmarks[36:42, 0], landmarks[36:42, 1], linestyle='-', color='red', lw=2)
    ax.plot([landmarks[41, 0], landmarks[36, 0]], [landmarks[41, 1], landmarks[36, 1]], linestyle='-', color='red', lw=2)
    
    ax.plot(landmarks[42:48, 0], landmarks[42:48, 1], linestyle='-', color='red', lw=2)
    ax.plot([landmarks[47, 0], landmarks[42, 0]], [landmarks[47, 1], landmarks[42, 1]], linestyle='-', color='red', lw=2)
    
    # Mouth
    ax.plot(landmarks[48:60, 0], landmarks[48:60, 1], linestyle='-', color='purple', lw=2)
    ax.plot([landmarks[59, 0], landmarks[48, 0]], [landmarks[59, 1], landmarks[48, 1]], linestyle='-', color='purple', lw=2)
    
    fig.canvas.draw()
    
    data = Image.frombuffer('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(), 'raw', 'RGB', 0, 1)
    
    plt.close(fig)
    
    return data

def lm_adapt(target_lm, user_lm):

    # Scale invariance
    tarHeadWidth = target_lm[16, 0] - target_lm[0, 0]
    tarHeadHeight = target_lm[8, 1] - target_lm[0, 1]
    
    usrHeadWidth = user_lm[16, 0] - user_lm[0, 0]
    usrHeadHeight = user_lm[8, 1] - user_lm[0, 1]

    # Eyebrow to eye adaptation
    tarEyeBrow = target_lm[17:27, 1]
    tarEye = target_lm[36:48, 1]
    
    tarEyeToEyeBrow = np.true_divide(np.subtract(tarEye.mean(), tarEyeBrow.mean()), tarHeadHeight) # portion of eye-erebrow
    
    usrEyeBrow = user_lm[17:27, 1]
    usrEye = user_lm[36:48, 1]
    
    usrEyeToEyeBrow = np.true_divide(np.subtract(usrEye.mean(), usrEyeBrow.mean()), usrHeadHeight)
    
    dif_EyeToEyeBrow = np.subtract(usrEyeToEyeBrow, tarEyeToEyeBrow) * usrHeadHeight
    
    adaptedEyeBrow_Y = np.expand_dims(usrEyeBrow + dif_EyeToEyeBrow * 0.8, axis=1)
    
    # Eye to eye adaptation
    tarLeye = target_lm[36:42, 0]
    tarReye = target_lm[42:48, 0]
    
    tarEyeToEye = np.true_divide(tarLeye.mean() - tarReye.mean(), tarHeadWidth)
    
    usrLeye = user_lm[36:42, 0]
    usrReye = user_lm[42:48, 0]
    
    usrEyeToEye = np.true_divide(usrLeye.mean() - usrReye.mean(), usrHeadWidth)
    
    dif_EyeToEye = (usrEyeToEye - tarEyeToEye) * usrHeadWidth
    
    adaptedLeye = np.expand_dims(np.subtract(usrLeye, dif_EyeToEye/3.0), axis=1)
    adaptedReye = np.expand_dims(np.add(usrReye, dif_EyeToEye/3.0), axis=1)
    
    adaptedEyeBrow = np.concatenate((user_lm[17:27,:-1], adaptedEyeBrow_Y.round()), axis=1)
    adaptedEye = np.concatenate((np.concatenate((adaptedLeye, adaptedReye)).round(), np.expand_dims(usrEye,axis=1)), axis=1)
    
    #print(usrHeadHeight)
    #print(usrHeadWidth)
    
    adapted_lm = np.concatenate((user_lm[:17], adaptedEyeBrow, user_lm[27:36], adaptedEye, user_lm[48:]))
    
    return adapted_lm
    
def preProcess(img, tar_lm):

    EyeCenter = tar_lm[36:48,1].mean()
    Chin = tar_lm[8,1]
    Chin_X = tar_lm[8,0]
    headCenter_X = tar_lm[0:17, 0].mean()
    
    center = (EyeCenter + Chin) /2

    face_ratio = 110.0 / (Chin - EyeCenter)
    
    Top = int(EyeCenter - (69 / face_ratio))

    size = int(128 / face_ratio)
    
    #transform= transforms.Resize(size)

    #img = transform(img)

    img = TF.resized_crop(img, Top, headCenter_X-size, 2*size, 2*size, size=256)
    return img

def preProcess2(tar, tar_lm):
   
    EyeCenter = tar_lm[36:48,1].mean()
    Chin = tar_lm[8,1]
    Chin_X = tar_lm[8,0]
    headCenter_X = tar_lm[0:17, 0].mean()
    
    center = (EyeCenter + Chin) /2

    face_ratio = 110.0 / (Chin - EyeCenter)
    
    Top = int(EyeCenter - (69 / face_ratio))

    size = int(128 / face_ratio)
    
    #transform= transforms.Resize(size)

    #img = transform(img)

    tar = tar[int(Top): int(Top+2*size), int(headCenter_X-size) : int(headCenter_X-size+2*size),:] #TF.resized_crop(tar, Top, headCenter_X-size, 2*size, 2*size, size=256)
    return tar