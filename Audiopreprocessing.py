import torch
import torchaudio
from glob import glob
from PIL import Image
import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import torch.nn as nn
import os


def make_spectrogram(path):
    SAMPLE_RATE=8000
    WIN_LENGTH=None
    N_FFT=512
    HOP_SIZE=N_FFT//2
    N_MELS=128
    
    for s in glob(f"{path}/*.wav"):
        data,sr = torchaudio.load(s)
        transform = nn.Sequential(
            torchaudio.transforms.Resample(orig_freq=sr, new_freq= SAMPLE_RATE),
            torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,win_length=WIN_LENGTH,n_fft=N_FFT,normalized=True,n_mels=N_MELS,hop_length=HOP_SIZE),
            torchaudio.transforms.AmplitudeToDB(),
        )
        # 모노채널로 변환
        data=data.mean(dim=0)
        
        
        # Transform 수행
        t_data = transform(data)
        min_value = np.min((t_data.numpy()))
        max_value = np.max((t_data.numpy()))

        # Sepctrogram 저장
        output = (t_data.numpy() - min_value) / (max_value - min_value)
        output = Image.fromarray((255*output).astype("uint8"))
        
        name=s.split("/")[-1].split(".")[0]
        output.save(f"{path}/{name}.png")
        
        
def split_spectrogram(path):
    durations={}
    for s in sorted(glob(f"{path}/*.*"),reverse=True):
        
        if s.split(".")[-1]=='wav':
            #sound
            data,sr = torchaudio.load(s)
            name=s.split("/")[-1].split(".")[0]
            durations[name] = len(data[0])/sr
            
        elif s.split(".")[-1]=='png':        
            #image
            name = s.split("/")[-1].split(".")[0]
            img = cv2.imread(s,0)
            H,W = img.shape
            win_size = np.ceil((W*0.5)/durations[name]).astype(np.int32)
            
            for idx,i in enumerate(range(0,len(img[0]),win_size)):
                new_img = img[:,i:i+win_size]
                os.makedirs(f"{path}/imgs",exist_ok=True)
                cv2.imwrite(f"{path}/imgs/{name}_{idx}.png",new_img)
if __name__=='__main__':
    PATH="./TTS/archive/kss"
    for i in range(4,5):
        make_spectrogram(PATH+f"/FAKE{i}")
        split_spectrogram(PATH+f"/FAKE{i}")
    


