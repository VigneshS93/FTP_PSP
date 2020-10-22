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
from models import ftp_psp, ftp_psp1, ftp_psp2, ftp_psp3, ftp_psp4, ftp_psp5
from torch.utils.data import DataLoader
from datas import dataset_loader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scd 
import sys
sys.path.append("..")
from utils.logutils import LogUtils
import utils.check_points_utils as checkpoint_util
from cannyEdge import CannyFilter

#Pass the arguments
parser = argparse.ArgumentParser(description="art_rem")
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
parser.add_argument("--data_dir", type=str, default=" ", help='path of data')
parser.add_argument("--log_dir", type=str, default=" ", help='path of log files')
parser.add_argument("--write_freq", type=int, default=2, help="Step for saving Checkpoint")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to start from")
parser.add_argument("--co_eff", type=float, default=0.7, help="Coefficient for loss")
parser.add_argument("--low_th", type=float, default=0, help="Lower threshold for canny")
parser.add_argument("--high_th", type=float, default=0.2, help="Higher threshold for canny")
parser.add_argument("--gpu_no", type=str, default="5", help="GPU number")

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_no

# load the testing data set
input_set, groundTruth_set,mask, filenames = dataset_loader(opt.data_dir)
input_set = np.array(input_set, dtype=np.float32)
groundTruth_set = np.array(groundTruth_set, dtype=np.float32)
mask = torch.FloatTensor(np.array(mask))
mask = mask/255
test_set=[]
for i in range(len(input_set)):
  test_set.append([input_set[i], groundTruth_set[i], mask[i], filenames[i]])
testLoader = DataLoader(dataset=test_set, num_workers=0, batch_size=opt.batchSize, shuffle=True, pin_memory=True)

# Define the loss function
# mse_loss = nn.MSELoss(reduction='mean')
canny_edge = CannyFilter().cuda()
def squared_diff(mask, output, groundTruth):
  sq_diff = torch.square(output - groundTruth)
  mask_sq_diff = torch.mul(mask, sq_diff)
  loss = torch.mean(mask_sq_diff)
  return loss
def edgeLoss(mask, output_edge, groundTruth):
  # output_edge = canny_edge(output, opt.low_th, opt.high_th)
  gt_edge = canny_edge(groundTruth, opt.low_th, opt.high_th)
  diff = torch.abs(output_edge - gt_edge)
  mask_diff = torch.mul(mask, diff)
  loss = mask_diff.float().mean()
  return loss
def mean_abs_loss(mask, output, groundTruth):
  diff = torch.abs(output - groundTruth)
  mask_diff = torch.mul(mask, diff)
  loss = torch.mean(mask_diff)
  return loss
def mse_edge_loss(mask, output, out_edge, groundTruth, co_eff):
  # mse_loss = squared_diff(mask, output, groundTruth)
  mae_loss = mean_abs_loss(mask, output, groundTruth)
  edge_loss = edgeLoss(mask, out_edge, groundTruth)
  loss = co_eff*mae_loss + (1-co_eff)*edge_loss
  return loss, edge_loss


#Define the log directory for checkpoints
if os.path.exists(opt.log_dir) is not True:
  os.makedirs(opt.log_dir)

checkpoints_dir = os.path.join(opt.log_dir, "checkpoints")

if os.path.exists(checkpoints_dir) is not True:
  os.mkdir(checkpoints_dir)

# Load the model
input_channel=1
model = ftp_psp5(input_channel).cuda()
# model = nn.DataParallel(model) # For using multiple GPUs

#Load status from checkpoint 
log_open_mode = 'w'
checkpoint_util.load_checkpoint(model_3d=model, filename=opt.checkpoint)

log = LogUtils(os.path.join(opt.log_dir, 'logfile'), log_open_mode)
log.write('Supervised learning for phase map enhancement - Testing\n')
log.write_args(opt)

if opt.checkpoint is None:
    print('Checkpoint is missing! Load the checkpoint to start the testing')
    sys.exit()
    
# Test the network using the trained model
testData = iter(testLoader)
ave_loss = 0
count = 0
for data in iter(testLoader):
    inp_PM, gt_PM, mask_PM, filename_PM = next(testData)
    inp_PM = torch.unsqueeze(inp_PM,1).cuda()
    gt_PM = torch.unsqueeze(gt_PM,1).cuda()
    mask_PM = torch.unsqueeze(mask_PM,1).cuda()
    output_PM, out_edge_PM = model(inp_PM)
    loss, edge_loss = mse_edge_loss(mask_PM, output_PM, out_edge_PM, gt_PM, opt.co_eff)
    ave_loss += loss.item()

    # Write CSV files
    inp = inp_PM[0][0].detach().cpu().numpy()
    filename = opt.log_dir + str("/batch_") + str(count) + str("_inputPM.csv")
    pd.DataFrame(inp).to_csv(filename,header=False,index=False)
    out = output_PM[0][0].detach().cpu().numpy()
    filename = opt.log_dir + str("/batch_") + str(count) + str("_outputPM.csv")
    pd.DataFrame(out).to_csv(filename,header=False,index=False)
    gt = gt_PM[0][0].detach().cpu().numpy()
    filename = opt.log_dir + str("/batch_") + str(count) + str("_gtPM.csv")
    pd.DataFrame(gt).to_csv(filename,header=False,index=False)
    # Write down the filename
    f_name = str.encode(filename_PM[0])
    filename = opt.log_dir + str("/batch_") + str(count) + str("fname.csv")
    with open(filename,"wb") as file:
      file.write(f_name)

    count += 1
ave_loss /= count


# Log the results
log.write('\nAverage_test_loss:{0}'.format(("%.8f" % ave_loss)))


print('Finished Testing') 


