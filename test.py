import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from matplotlib.pyplot import imread
from models import art_rem
from torch.utils.data import DataLoader
from datas import dataset_loader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scd 
import sys
sys.path.append("..")
from utils.logutils import LogUtils
import utils.check_points_utils as checkpoint_util
# from dataloader import load_data as data_loader


#Pass the arguments
parser = argparse.ArgumentParser(description="art_rem")
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
parser.add_argument("--data_dir", type=str, default=" ", help='path of data')
parser.add_argument("--log_dir", type=str, default=" ", help='path of log files')
parser.add_argument("--write_freq", type=int, default=2, help="Step for saving Checkpoint")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to start from")

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# load the training data set
input_set, groundTruth_set = dataset_loader(opt.data_dir)
input_set = np.array(input_set, dtype=np.float32)
groundTruth_set = np.array(groundTruth_set, dtype=np.float32)
test_set=[]
for i in range(len(input_set)):
  test_set.append([input_set[i], groundTruth_set[i]])
testLoader = DataLoader(dataset=test_set, num_workers=0, batch_size=opt.batchSize, shuffle=True, pin_memory=True)

# Define the loss function
mse_loss = nn.MSELoss()

#Define the log directory for checkpoints
if os.path.exists(opt.log_dir) is not True:
  os.makedirs(opt.log_dir)

checkpoints_dir = os.path.join(opt.log_dir, "checkpoints")

if os.path.exists(checkpoints_dir) is not True:
  os.mkdir(checkpoints_dir)

# Load the model
input_channel=1
model = art_rem(input_channel).cuda()
# model = nn.DataParallel(model) # For using multiple GPUs

#Load status from checkpoint 
log_open_mode = 'w'
checkpoint_util.load_checkpoint(model_3d=model, filename=opt.checkpoint)

log = LogUtils(os.path.join(opt.log_dir, 'logfile'), log_open_mode)
log.write('Supervised learning for motion artifact reduction - Testing\n')
log.write_args(opt)

if opt.checkpoint is None:
    print('Checkpoint is missing! Load the checkpoint to start the testing')
    sys.exit()
    
# Test the network using the trained model
testData = iter(testLoader)
ave_loss = 0
count = 0
for data in iter(testLoader):
    inp_PM, gt_PM = next(testData)
    inp_PM = torch.unsqueeze(inp_PM,1).cuda()
    gt_PM = torch.unsqueeze(gt_PM,1).cuda()
    output_PM = model(inp_PM)
    loss = mse_loss(output_PM, gt_PM)
    ave_loss += loss.item()
    count += 1
ave_loss /= count

# Write CSV files
inp = inp_PM[0][0].detach().cpu().numpy()
filename = opt.log_dir + str("/epoch_") + str(count) + str("_inputPM.csv")
pd.DataFrame(inp).to_csv(filename,header=False,index=False)
out = output_PM[0][0].detach().cpu().numpy()
filename = opt.log_dir + str("/epoch_") + str(count) + str("_outputPM.csv")
pd.DataFrame(out).to_csv(filename,header=False,index=False)

# Log the results
log.write('\nAverage_test_loss:{0}'.format(("%.8f" % ave_loss)))


print('Finished Testing') 


