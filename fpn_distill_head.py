
import torch.nn as nn

class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)



class FPNHead(nn.Module):
    

   """ The FPN gives us 4 feature maps as inputs: 
       { (1, 256, 32, 32),
         (1, 256, 16, 16),
         (1, 256, 8, 8),
         (1, 256, 4, 4)
   }"""
   
   def __init__(self, num_classes=100, n_maps=5):
       super(FPNHead, self).__init__()
   
       # Everything needs to be mapped to (256,2,2)
       self.pool = nn.AdaptiveAvgPool2d((2,2))
       self.fcs = nn.ModuleList([nn.Linear(1024, num_classes) for i in range(n_maps)]) 
   
   def forward(self, x):
       maps = []
       outs = []
   
       for i in x:
          maps.append(self.pool(i).view(i.size(0), -1))
   
       for map_ in range(len(maps)):
          outs.append(self.fcs[map_](maps[map_]))
   
       return outs
