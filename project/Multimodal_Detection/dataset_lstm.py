import torch
from torch.utils import data
from torchvision import transforms
import os
import sys
import time
import pickle
import glob
import csv
import pandas as pd
import numpy as np
import cv2
from augmentation import *
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.io import wavfile
import python_speech_features

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class deepfake_3d(data.Dataset):
    def __init__(self,out_dir,
                 mode='train',
                 transform=None):
        self.mode = mode
        self.transform = transform
        self.out_dir = out_dir

        # splits
        if mode == 'train':
            split = os.path.join(self.out_dir,'train_split_lstm.csv')
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'test'):
            split = os.path.join(self.out_dir,'test_split_lstm.csv')
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # get label list
        self.label_dict_encode = {}
        self.label_dict_decode = {}
        self.label_dict_encode['fake'] = 0
        self.label_dict_decode['0'] = 'fake'
        self.label_dict_encode['real'] = 1
        self.label_dict_decode['1'] = 'real'

        self.video_info = video_info

    def __getitem__(self, index):
        vpath,label = self.video_info.iloc[index]
        segpath = [] #vpath = ['path/00000','path/00001' ....
        audiopath = []
        for seg in os.listdir(vpath):
            if seg.split('.')[-1] != 'wav':
                segpath.append(os.path.join(vpath, seg))
            else:
                audiopath.append(os.path.join(vpath, seg))
        frames_all_seg = []
        for each_seg in segpath:
            frames_per_seg = [pil_loader(os.path.join(each_seg,frame)) for frame in os.listdir(each_seg)]
            frames_all_seg = frames_all_seg + frames_per_seg
        t_frames_all_seg = self.transform(frames_all_seg)
        (C, H, W) = t_frames_all_seg[0].size()
        t_frames_all_seg = torch.stack(t_frames_all_seg, 0)
        t_frames_all_seg =  t_frames_all_seg.view(-1, 30, C, H, W).transpose(1, 2)

        cc_all = []
        for audiopath_per_seg in audiopath:
            sample_rate, audio = wavfile.read(audiopath_per_seg)

            mfcc = zip(*python_speech_features.mfcc(audio,sample_rate,nfft=2048))
            mfcc = np.stack([np.array(i) for i in mfcc])
            cc = np.expand_dims(mfcc,axis=0)
            cc_all.append(cc)
        cc_allnp = np.stack([i for i in cc_all])
        cct = torch.autograd.Variable(torch.from_numpy(cc_allnp.astype(float)).float())

        vid = self.encode_label(label)

        return t_frames_all_seg, cct, torch.LongTensor([vid]),vpath

    def __len__(self):
        return len(self.video_info)

    def encode_label(self, label_name):
        return self.label_dict_encode[label_name]

    def decode_label(self, label_code):
        return self.label_dict_decode[label_code]