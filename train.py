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
from models import ftp_psp, ftp_psp1, ftp_psp2, ftp_psp3
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
import canny_edge_detector as canny
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
parser.add_argument("--gpu_no", type=str, default="0", help="path of log files")

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
mse_loss = nn.MSELoss(reduction='mean')
def squared_diff(mask, output, groundTruth):
  sq_diff = torch.square(output - groundTruth)
  mask_sq_diff = torch.mul(mask,sq_diff)
  loss = torch.sqrt(torch.mean(mask_sq_diff))
  return loss
# edg = canny.cannyEdgeDetector(output,sigma=2,kernel_size=5,lowthreshold=0.09,highthreshold=0.17,weak_pixel=50)
def edge_loss(out, target, cuda=True):
	x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
	y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	convx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
	convy = nn.Conv2d(1, 1, kernel_size=3 , stride=1, padding=1, bias=False)
	weights_x = torch.from_numpy(x_filter).float().unsqueeze(0).unsqueeze(0)
	weights_y = torch.from_numpy(y_filter).float().unsqueeze(0).unsqueeze(0)

	if cuda:
		weights_x = weights_x.cuda()
		weights_y = weights_y.cuda()

	convx.weight = nn.Parameter(weights_x)
	convy.weight = nn.Parameter(weights_y)

	g1_x = convx(out)
	g2_x = convx(target)
	g1_y = convy(out)
	g2_y = convy(target)

	g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2))
	g_2 = torch.sqrt(torch.pow(g2_x, 2) + torch.pow(g2_y, 2))

	return torch.mean((g_1 - g_2).pow(2))

def mse_edge_loss(mask,output,groundTruth):
  sq_diff = torch.square(output - groundTruth)
  mask_sq_diff = torch.mul(mask,sq_diff)
  mse_loss = torch.mean(mask_sq_diff)
  edg_loss = edge_loss(output,groundTruth)
  loss = 0.8*mse_loss+0.2*edg_loss
  return loss
iters = -1

#Define the log directory for checkpoints
if os.path.exists(opt.log_dir) is not True:
  os.makedirs(opt.log_dir)

checkpoints_dir = os.path.join(opt.log_dir, "checkpoints")

if os.path.exists(checkpoints_dir) is not True:
  os.mkdir(checkpoints_dir)

# Load the model
input_channel=1
model = ftp_psp2(input_channel).cuda()
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
    output_PM = model(inp_PM)
    # rg_loss = percept_loss(output_PM,reg)
    loss = squared_diff(mask_PM, output_PM, gt_PM)
    # loss = loss + rg_loss
    loss.backward()
    optimizer.step()
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


