import torch.nn as nn
import torch
import math

nc = 3
ndf = 64
class FusionDiscriminator(nn.Module):
    def __init__(self, input_nc, n_layer = 4, size=256, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(FusionDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

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
        for i in range(n_layer):
            prev_ndf = curr_ndf
            curr_ndf = min(prev_ndf * 2, ndf * 8)
            sequence += [
                nn.Conv2d(prev_ndf, curr_ndf, 4, 2, 1, bias=False),
                norm_layer(curr_ndf),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        self.shared = nn.Sequential(*sequence)

        # ===============
        # Real Layer
        # ===============
        sequence = []
        for i in range(n_layer, _iter):
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
        self.distortLayer = nn.Sequential(*sequence) # Classify distort or not, only output a scaler
    
        # ===============
        # Distort Layer
        # ===============
        sequence = []

        sequence += [
            nn.Conv2d(prev_ndf, curr_ndf, 4, 1, 0, bias=False),
            norm_layer(curr_ndf),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(curr_ndf, 1, 4, 1, 0, bias=False)]
        self.realLayer = nn.Sequential(*sequence) # Classify real or fake, output many patchs

    def forward(self, input):
        if input.is_cuda and len(self.gpu_ids) > 1:
            features = nn.parallel.data_parallel(self.shared, input, self.gpu_ids)
            distort = nn.parallel.data_parallel(self.distortLayer, features, self.gpu_ids).view(-1, 1)
            real = nn.parallel.data_parallel(self.realLayer, features, self.gpu_ids)
            return real, distort
        else:
            features = self.shared(input)
            print(features.shape)
            distort = self.distortLayer(features).view(-1, 1)
            real = self.realLayer(features)
            return real, distort


a = torch.zeros((4, 3, 256, 256))
D = FusionDiscriminator(3, 3, 256)
print(D.shared)
print(D.realLayer)
print(D.distortLayer)
o = D(a)
print(o[0].shape, o[1].shape)
