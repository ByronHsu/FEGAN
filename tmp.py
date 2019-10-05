import torch.nn as nn
import torch
import math

nc = 3
ndf = 64
class DCGANDiscriminator(nn.Module):
    def __init__(self, input_nc, size=256, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(DCGANDiscriminator, self).__init__()
        self.ngpu = gpu_ids

        sequence = []
        sequence += [
            # input is (nc) x size x size
            nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # now state is ndf x (size // 2) x (size // 2)
        # we need to downsize (size // 2) x (size // 2) to 4 x 4
        _iter = int(math.log((size // 8), 2)) # calculate how mamy times needed to iterate
        prev_ndf = ndf
        curr_ndf = ndf
        for i in range(_iter):
            prev_ndf = curr_ndf
            curr_ndf = min(prev_ndf * 2, ndf * 8)
            sequence += [
                nn.Conv2d(prev_ndf, curr_ndf, 4, 2, 1, bias=False),
                norm_layer(curr_ndf),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # now state is _ x 4 x 4
        # this layer will downsize it to 1 x 1
        sequence += [
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        ]
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


a = torch.zeros((4, 3, 256, 256))
D = DCGANDiscriminator(3, 256)
print(D.main)
print(D(a).shape)
