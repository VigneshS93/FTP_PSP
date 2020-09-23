#model
import torch
import torch.nn as nn
import torch.nn.functional as F

class art_rem(torch.nn.Module):
    def __init__(self, inp_ch):
        super(art_rem, self).__init__()
        ks = 3
        pad = 1
        out_ch_1 = 10
        out_ch_2 = 20
        out_ch_3 = 30
        self.conv1 = nn.Conv2d(inp_ch, out_ch_1, kernel_size=ks, stride=1, padding=pad)
        self.bnorm1 = nn.BatchNorm2d(out_ch_1)
        self.mpool1 = nn.MaxPool2d(2, stride=2,return_indices=True)
        self.conv2 = nn.Conv2d(out_ch_1, out_ch_1, kernel_size=ks, stride=1, padding=pad)
        self.bnorm2 = nn.BatchNorm2d(out_ch_1)
        self.mpool2 = nn.MaxUnpool2d(2, stride=2)
        self.conv3 = nn.Conv2d(out_ch_1, inp_ch, kernel_size=ks, stride=1, padding=pad)
        self.bnorm3 = nn.BatchNorm2d(inp_ch)
   
    def forward(self, X):
        h = F.relu(self.bnorm1(self.conv1(X)))
        # h, indices = self.mpool1(h)
        # h = F.relu(h)
        h = F.relu(self.bnorm2(self.conv2(h)))
        # h = F.relu(self.mpool2(h, indices))
        h = (self.conv3(h))
        
        return h

class art_rem1(torch.nn.Module):
    def __init__(self, inp_ch):
        super(art_rem1, self).__init__()
        ks = 3
        pad = 1
        out_ch_1 = 10
        out_ch_2 = 20
        out_ch_3 = 30
        self.conv1 = nn.Conv2d(inp_ch, out_ch_1, kernel_size=ks, stride=1, padding=pad)
        self.bnorm1 = nn.BatchNorm2d(out_ch_1)
        self.conv2 = nn.Conv2d(out_ch_1, out_ch_2, kernel_size=ks, stride=1, padding=pad)
        self.bnorm2 = nn.BatchNorm2d(out_ch_2)
        self.conv3 = nn.Conv2d(out_ch_2, out_ch_3, kernel_size=ks, stride=1, padding=pad)
        self.bnorm3 = nn.BatchNorm2d(out_ch_3)
        self.mpool1 = nn.MaxPool2d(2, stride=2,return_indices=True)
        self.conv4 = nn.Conv2d(out_ch_3, out_ch_3, kernel_size=ks, stride=1, padding=pad)
        self.bnorm4 = nn.BatchNorm2d(out_ch_3)
        self.conv5 = nn.Conv2d(out_ch_3, out_ch_3, kernel_size=ks, stride=1, padding=pad)
        self.bnorm5 = nn.BatchNorm2d(out_ch_3)
        self.mpool2 = nn.MaxUnpool2d(2, stride=2)
        self.conv6 = nn.Conv2d(out_ch_3, out_ch_2, kernel_size=ks, stride=1, padding=pad)
        self.bnorm6 = nn.BatchNorm2d(out_ch_2)
        self.conv7 = nn.Conv2d(out_ch_2, out_ch_1, kernel_size=ks, stride=1, padding=pad)
        self.bnorm7 = nn.BatchNorm2d(out_ch_1)
        self.conv8 = nn.Conv2d(out_ch_1, inp_ch, kernel_size=ks, stride=1, padding=pad)
        self.bnorm8 = nn.BatchNorm2d(inp_ch)
     
        
       
    def forward(self, X):
        h = F.relu(self.bnorm1(self.conv1(X)))
        h = F.relu(self.bnorm2(self.conv2(h)))
        h = F.relu(self.bnorm3(self.conv3(h)))
        # h = F.relu(self.conv3(h))
        h, indices = self.mpool1(h)
        h = F.relu(h)
        h = F.relu(self.bnorm4(self.conv4(h)))
        h = F.relu(self.bnorm5(self.conv5(h)))
        h = F.relu(self.mpool2(h, indices))
        h = F.relu(self.bnorm6(self.conv6(h)))
        h = F.relu(self.bnorm7(self.conv7(h)))
        h = (self.conv8(h))
        
        return h