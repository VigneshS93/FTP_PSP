#model
import torch
import torch.nn as nn
import torch.nn.functional as F

class ftp_psp(torch.nn.Module):
    def __init__(self, inp_ch):
        super(ftp_psp, self).__init__()
        ks = 3
        pad = 1
        out_ch = 10
        self.conv1 = nn.Conv2d(inp_ch, inp_ch, kernel_size=ks, stride=1, padding=pad)
        self.conv2 = nn.Conv2d(inp_ch, out_ch, kernel_size=ks, stride=1, padding=pad)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=ks, stride=1, padding=pad)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.conv4 = nn.Conv2d(out_ch, inp_ch, kernel_size=ks, stride=1, padding=pad)
        self.bnorm3 = nn.BatchNorm2d(inp_ch)
        self.conv5 = nn.Conv2d(inp_ch, inp_ch, kernel_size=ks, stride=1, padding=pad)
        
       
    def forward(self, X):
        h = self.conv1(X)
        h = F.relu((self.conv2(h)))
        h = F.relu((self.conv3(h)))
        h = F.relu((self.conv4(h)))
        h = self.conv5(h)
        # h = h + X
        
        return h

class ftp_psp1(torch.nn.Module):
    def __init__(self, inp_ch):
        super(ftp_psp1, self).__init__()
        ks = 3
        pad = 1
        out_ch1 = 10
        out_ch2 = 20
        self.conv1 = nn.Conv2d(inp_ch, out_ch1, kernel_size=ks, stride=1, padding=pad)
        self.bnorm1 = nn.BatchNorm2d(out_ch1)
        self.conv2 = nn.Conv2d(out_ch1, out_ch1, kernel_size=ks, stride=1, padding=pad)
        self.bnorm2 = nn.BatchNorm2d(out_ch1)
        self.conv3 = nn.Conv2d(out_ch1, inp_ch, kernel_size=ks, stride=1, padding=pad)
        self.conv4 = nn.Conv2d(inp_ch, out_ch1, kernel_size=ks, stride=1, padding=pad)
        self.conv5 = nn.Conv2d(out_ch1, out_ch2, kernel_size=ks, stride=1, padding=pad)
        self.conv6 = nn.Conv2d(out_ch2, out_ch1, kernel_size=ks, stride=1, padding=pad)
        self.conv7 = nn.Conv2d(out_ch1, inp_ch, kernel_size=ks, stride=1, padding=pad)
        
    def forward(self, X):
        h = F.relu((self.conv1(X)))
        h = F.relu((self.conv2(h)))
        h = self.conv3(h)
        # h = h + X
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = self.conv7(h)
        return h

class ftp_psp2(torch.nn.Module):
    def __init__(self, inp_ch):
        super(ftp_psp2, self).__init__()
        ks = 3
        pad = 1
        out_ch1 = 10
        out_ch2 = 20
        out_ch3 = 30
        out_ch4 = 40
        self.conv1 = nn.Conv2d(inp_ch, out_ch1, kernel_size=ks, stride=1, padding=pad)
        self.conv2 = nn.Conv2d(out_ch1, out_ch2, kernel_size=ks, stride=1, padding=pad)
        self.conv3 = nn.Conv2d(out_ch2, out_ch3, kernel_size=ks, stride=1, padding=pad)
        self.conv4 = nn.Conv2d(out_ch3, out_ch4, kernel_size=ks, stride=1, padding=pad)
        self.conv5 = nn.Conv2d(out_ch4, out_ch3, kernel_size=ks, stride=1, padding=pad)
        self.conv6 = nn.Conv2d(out_ch3, out_ch2, kernel_size=ks, stride=1, padding=pad)
        self.conv7 = nn.Conv2d(out_ch2, out_ch1, kernel_size=ks, stride=1, padding=pad)
        self.conv8 = nn.Conv2d(out_ch1, inp_ch, kernel_size=ks, stride=1, padding=pad)
        
    def forward(self, X):
        h = F.relu(self.conv1(X))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = self.conv7(h)
        h = self.conv8(h)
        
        return h

class ftp_psp3(torch.nn.Module):
    def __init__(self, inp_ch):
        super(ftp_psp3, self).__init__()
        ks = 3
        pad = 1
        out_ch1 = 10
        out_ch2 = 20
        out_ch3 = 30
        out_ch4 = 40
        out_ch5 = 50
        self.conv1 = nn.Conv2d(inp_ch, out_ch1, kernel_size=ks, stride=1, padding=pad)
        self.conv2 = nn.Conv2d(out_ch1, out_ch2, kernel_size=ks, stride=1, padding=pad)
        self.conv3 = nn.Conv2d(out_ch2, out_ch3, kernel_size=ks, stride=1, padding=pad)
        self.conv4 = nn.Conv2d(out_ch3, out_ch4, kernel_size=ks, stride=1, padding=pad)
        self.conv5 = nn.Conv2d(out_ch4, out_ch5, kernel_size=ks, stride=1, padding=pad)
        self.conv6 = nn.Conv2d(out_ch5, out_ch4, kernel_size=ks, stride=1, padding=pad)
        self.conv7 = nn.Conv2d(out_ch4, out_ch3, kernel_size=ks, stride=1, padding=pad)
        self.conv8 = nn.Conv2d(out_ch3, out_ch2, kernel_size=ks, stride=1, padding=pad)
        self.conv9 = nn.Conv2d(out_ch2, out_ch1, kernel_size=ks, stride=1, padding=pad)
        self.conv10 = nn.Conv2d(out_ch1, inp_ch, kernel_size=ks, stride=1, padding=pad)
        
    def forward(self, X):
        h = F.relu(self.conv1(X))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = self.conv9(h)
        h = self.conv10(h)
        
        return h
        
class ftp_psp4(torch.nn.Module):
    def __init__(self, inp_ch):
        super(ftp_psp4, self).__init__()
        ks = 3
        pad = 1
        out_ch1 = 10
        out_ch2 = 20
        out_ch3 = 30
        out_ch4 = 40
        self.conv1 = nn.Conv2d(inp_ch, out_ch1, kernel_size=ks, stride=1, padding=pad)
        self.conv2 = nn.Conv2d(out_ch1, out_ch2, kernel_size=ks, stride=1, padding=pad)
        self.conv3 = nn.Conv2d(out_ch2, out_ch3, kernel_size=ks, stride=1, padding=pad)
        self.conv4 = nn.Conv2d(out_ch3, out_ch4, kernel_size=ks, stride=1, padding=pad)
        self.conv5 = nn.Conv2d(out_ch4, out_ch3, kernel_size=ks, stride=1, padding=pad)
        self.conv6 = nn.Conv2d(out_ch3, out_ch2, kernel_size=ks, stride=1, padding=pad)
        self.conv7 = nn.Conv2d(out_ch2, out_ch1, kernel_size=ks, stride=1, padding=pad)
        self.conv8 = nn.Conv2d(out_ch1, inp_ch, kernel_size=ks, stride=1, padding=pad)
        
    def forward(self, X):
        h = F.relu(self.conv1(X))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h_edge = F.relu(self.conv6(h))
        h = self.conv7(h_edge)
        h = self.conv8(h)
        
        return h, h_edge

class ftp_psp5(torch.nn.Module):
    def __init__(self, inp_ch):
        super(ftp_psp5, self).__init__()
        ks = 3
        pad = 1
        out_ch1 = 10
        out_ch2 = 20
        out_ch3 = 30
        out_ch4 = 40
        self.conv1 = nn.Conv2d(inp_ch, out_ch1, kernel_size=ks, stride=1, padding=pad)
        self.conv2 = nn.Conv2d(out_ch1, out_ch2, kernel_size=ks, stride=1, padding=pad)
        self.conv3 = nn.Conv2d(out_ch2, out_ch3, kernel_size=ks, stride=1, padding=pad)
        self.conv4 = nn.Conv2d(out_ch3, out_ch4, kernel_size=ks, stride=1, padding=pad)
        self.conv5 = nn.Conv2d(out_ch4, out_ch3, kernel_size=ks, stride=1, padding=pad)
        self.conv6 = nn.Conv2d(out_ch3, inp_ch, kernel_size=ks, stride=1, padding=pad)
        self.conv7 = nn.Conv2d(inp_ch, out_ch2, kernel_size=ks, stride=1, padding=pad)
        self.conv8 = nn.Conv2d(out_ch2, out_ch1, kernel_size=ks, stride=1, padding=pad)
        self.conv9 = nn.Conv2d(out_ch1, inp_ch, kernel_size=ks, stride=1, padding=pad)
        
    def forward(self, X):
        h = F.relu(self.conv1(X))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h_edge = F.relu(self.conv6(h))
        h = self.conv7(h_edge)
        h = self.conv8(h)
        h = self.conv9(h)
        
        return h, h_edge