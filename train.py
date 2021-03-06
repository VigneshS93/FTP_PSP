"""
Created on Sep 23
@Author: Vignesh Suresh
"""
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
from torch.autograd import Variable
from torchvision import transforms
from datas import normalizeData
from cannyEdge import CannyFilter
# from dataloader import load_data as data_loader


#Pass the arguments
parser = argparse.ArgumentParser(description="ftp_psp2")
parser.add_argument("--batchSize", type=int, default=4, help="Training batch size")
parser.add_argument("--num_epochs", type=int, default=600, help="Number of training epochs")
parser.add_argument("--decay_step", type=int, default=1000, help="The step at which the learning rate should drop")
parser.add_argument("--lr_decay", type=float, default=0.5, help="Rate at which the learning rate should drop")
parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
parser.add_argument("--data_dir", type=str, default=" ", help="path of data")
parser.add_argument("--log_dir", type=str, default=" ", help="path of log files")
parser.add_argument("--write_freq", type=int, default=10, help="Step for saving Checkpoint")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to start from")
parser.add_argument("--gpu_no", type=str, default="0", help="GPU number")
parser.add_argument("--low_th", type=float, default=0, help="Lower threshold for canny")
parser.add_argument("--high_th", type=float, default=0.2, help="Higher threshold for canny")
parser.add_argument("--co_eff", type=float, default=0.7, help="Coefficient for loss")

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_no

# load the training data set
input_set, groundTruth_set, mask, filenames = dataset_loader(opt.data_dir)
input_set = torch.FloatTensor(np.array(input_set))
groundTruth_set = torch.FloatTensor(np.array(groundTruth_set))
mask = torch.FloatTensor(np.array(mask))
mask = mask/255
# norm_input = normalizeData(input_set)
# norm_gt = normalizeData(groundTruth_set)
train_set=[]
for i in range(len(input_set)):
  # train_set.append([norm_input[i], norm_gt[i], mask[i], filenames[i]])
  train_set.append([input_set[i], groundTruth_set[i], mask[i], filenames[i]])
trainLoader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batchSize, shuffle=True, pin_memory=True)

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

iters = -1

#Define the log directory for checkpoints
if os.path.exists(opt.log_dir) is not True:
  os.makedirs(opt.log_dir)

checkpoints_dir = os.path.join(opt.log_dir, "checkpoints")

if os.path.exists(checkpoints_dir) is not True:
  os.mkdir(checkpoints_dir)

# Load the model
input_channel=1
model = ftp_psp5(input_channel).cuda()
model = nn.DataParallel(model) # For using multiple GPUs

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

#Load status from checkpoint 
log_open_mode = 'w'
start_epoch = 0
if opt.checkpoint is not None:
    fname = os.path.join(checkpoints_dir, opt.checkpoint)
    start_epoch, iters = checkpoint_util.load_checkpoint(model_3d=model, optimizer=optimizer, filename=fname)
    start_epoch += 1
    log_open_mode = 'a'

log = LogUtils(os.path.join(opt.log_dir, 'logfile'), log_open_mode)
log.write('Supervised learning for phase map enhancement - Training\n')
log.write_args(opt)
lr_scheduler = lr_scd.StepLR(optimizer, step_size=opt.decay_step, gamma=opt.lr_decay)
iters = max(iters,0)
reg = 1e-7
# Train the network on the training dataset
for epoch_num in range(start_epoch, opt.num_epochs):
  trainData = iter(trainLoader)
  ave_loss = 0
  count = 0
  for data in iter(trainLoader):
    if lr_scheduler is not None:
      lr_scheduler.step(iters)
    optimizer.zero_grad()
    inp_PM, gt_PM, mask_PM, filename_PM = next(trainData)
    inp_PM = torch.unsqueeze(inp_PM,1).cuda()
    gt_PM = torch.unsqueeze(gt_PM,1).cuda()
    mask_PM = torch.unsqueeze(mask_PM,1).cuda()
    output_PM, out_edge_PM = model(inp_PM)
    loss, edge_loss = mse_edge_loss(mask_PM, output_PM, out_edge_PM, gt_PM, opt.co_eff)
    # loss = squared_diff(mask_PM, output_PM, gt_PM)
    loss.backward()
    optimizer.step()
    if count % 90 is 0:
      print('\nThe edge loss is %f.' %(edge_loss))
    iters += 1
    ave_loss += loss
    count += 1
  lr_scheduler.get_last_lr()
  ave_loss /= count
  for param_group in optimizer.param_groups:
    print('\nTraining at Epoch %d with a learning rate of %f.' %(epoch_num, param_group["lr"]))
  if opt.write_freq != -1 and (epoch_num + 1) % opt.write_freq is 0:
    fname = os.path.join(checkpoints_dir, 'checkpoint_{}'.format(epoch_num))
    checkpoint_util.save_checkpoint(filename=fname, model_3d=model, optimizer=optimizer, iters=iters, epoch=epoch_num)

  # Write CSV files
  inp = inp_PM[0][0].detach().cpu().numpy()
  filename = opt.log_dir + str("/epoch_") + str(epoch_num) + str("_inputPM.csv")
  pd.DataFrame(inp).to_csv(filename,header=False,index=False)
  # inp = ori_inp[0].detach().cpu().numpy()
  # filename = opt.log_dir + str("/epoch_") + str(epoch_num) + str("_orig_inputPM.csv")
  # pd.DataFrame(inp).to_csv(filename,header=False,index=False)
  out = output_PM[0][0].detach().cpu().numpy()
  filename = opt.log_dir + str("/epoch_") + str(epoch_num) + str("_outputPM.csv")
  pd.DataFrame(out).to_csv(filename,header=False,index=False)
  gt = gt_PM[0][0].detach().cpu().numpy()
  filename = opt.log_dir + str("/epoch_") + str(epoch_num) + str("_gtPM.csv")
  pd.DataFrame(gt).to_csv(filename,header=False,index=False)
  # gt = ori_gt[0].detach().cpu().numpy()
  # filename = opt.log_dir + str("/epoch_") + str(epoch_num) + str("_ori_gtPM.csv")
  # pd.DataFrame(gt).to_csv(filename,header=False,index=False)
  
  # Write down the filename
  f_name = str.encode(filename_PM[0])
  filename = opt.log_dir + str("/epoch_") + str(epoch_num) + str("fname.csv")
  with open(filename,"wb") as file:
    file.write(f_name)
  # Log the results
  log.write('\nepoch no.: {0}, Average_train_loss:{1}'.format((epoch_num), ("%.8f" % ave_loss)))
  
print('Finished Training')


