import torch
from torchvision import transforms
import pandas as pd
import os
import glob
import os.path
import numpy as np
import random
import h5py
import cv2
import glob
import torch.utils.data as udata

def dataset_loader(data_dir):
      extension = 'csv'
      # Read the input 
      directory=data_dir+str("/input")
      os.chdir(directory)
      files = [i for i in glob.glob('*.{}'.format(extension))]
      files = sorted(files)
      filename = []
      train_data = []
      for f in files:
            inp = pd.read_csv(f,header=None)
            filename_temp = str(f)
            filename.append(filename_temp)
            train_data.append(inp)
      #Read the groundtruth data
      directory=data_dir+str("/groundTruth")
      os.chdir(directory)
      files = [i for i in glob.glob('*.{}'.format(extension))]
      files = sorted(files)
      gt_data = []
      for f in files:
            inp = pd.read_csv(f,header=None)
            gt_data.append(inp)
      #Read the mask
      directory=data_dir+str("/mask")
      os.chdir(directory)
      files = [i for i in glob.glob('*.{}'.format(extension))]
      files = sorted(files)
      mask_data = []
      for f in files:
            inp = pd.read_csv(f,header=None)
            mask_data.append(inp)
      return train_data, gt_data, mask_data, filename

def normalizeData(data):
      norm = torch.zeros(np.shape(data))
      for bs in range(len(data)):
            temp = data[bs]           
            norm_temp = temp - torch.min(temp)
            norm_temp = norm_temp/torch.max(norm_temp)
            norm[bs] = norm_temp
            # temp = data[bs]
            # data_range = torch.max(temp) - torch.min(temp)
            # norm_temp = (temp - torch.min(temp))/data_range
            # norm_temp = 2 * norm_temp - 1
      # norm = torch.FloatTensor(np.array(norm))
      return norm
    
