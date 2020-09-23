#model
import torch
import torch.nn as nn
import torch.nn.functional as F

class ftp_psp(torch.nn.Module):
    def __init__(self, inp_ch):
        super(art_rem1, self).__init__()
        ks = 3
        pad = 1
        out_ch = 10
        self.conv1 = nn.Conv2d(inp_ch, out_ch, kernel_size=ks, stride=1, padding=pad)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=ks, stride=1, padding=pad)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, inp_ch, kernel_size=ks, stride=1, padding=pad)
        self.bnorm3 = nn.BatchNorm2d(inp_ch)
        
       
    def forward(self, X):
        h = F.relu(self.bnorm1(self.conv1(X)))
        h = F.relu(self.bnorm2(self.conv2(h)))
        h = F.relu(self.bnorm3(self.conv3(h)))
        
        return h