import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_w = 112
img_h = 112
channel = 3
emb_size = 512

num_worker = 1
grad_cip = 5.
print_freq = 100
checkpoint = None

num_classes = 93431
num_samples = 5179510
DATA_DIR = 'data'

faces_ms1m_folder = 'data/ms1m-retinaface-t1'
path_imgidx = os.path.join(faces_ms1m_folder, 'train.idx')
path_imgrec = os.path.join(faces_ms1m_folder, 'train.rec')
IMG_DIR = 'data/images'

pickle_file = 'data/faces_ms1m_112x112.pickle'
