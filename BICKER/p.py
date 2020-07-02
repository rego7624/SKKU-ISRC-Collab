import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import network

B = 4
K = 4
channel = 3
width = 256
height = 256

E = network.Embedder()
G = network.Generator()

dummy = torch.randn(B, channel, width, height)

E.load_state_dict(torch.load('models/Embedder_20190819_1215.pth'))
G.load_state_dict(torch.load('models/Generator_20190819_1215.pth'))

torch.onnx.export(E,(dummy, dummy), "webcam_demo.onnx",verbose=True)#, operator_export_type=torch.onnx.OperatorExportTypes.ONNX)
