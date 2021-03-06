import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from util.quiver import plot_quiver
from .base_model import BaseModel
from . import networks
import random
import math
import sys
import pdb
import torch.nn.functional as F
import torchvision.transforms as transforms 
import random
from PIL import Image
from skimage import io, feature, transform
import cv2

torch.autograd.set_detect_anomaly(True)
class GcGANCrossModel(BaseModel):
    def name(self):
        return 'GcGANCrossModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        flow_nc = 2

        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)


        self.netG_AB = networks.define_G(opt.input_nc, flow_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.use_att, self.gpu_ids)
        self.netG_gc_AB = self.netG_AB # share G_gc and G

        
        self.true = torch.ones((nb, 1)).cuda()
        self.false = torch.zeros((nb, 1)).cuda()
        
        # read chessboard
        img_path = os.path.join('./util/chessboard.jpg')
        img = Image.open(img_path)
        
        transform1 = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor() # range [0, 255] -> [0.0,1.0]
            ]
        )
        self.chess = transform1(img).unsqueeze(0).cuda()

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, size,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt.use_att)
            self.netD_gc_B = self.netD_B  # share D_B and D_gc_B

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_AB, 'G_AB', which_epoch)
            self.load_network(self.netG_gc_AB, 'G_AB', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_B, 'D_B', which_epoch)
                self.load_network(self.netD_gc_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_gc_B_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionGc = torch.nn.L1Loss()
            self.criterionCrossFlow = torch.nn.L1Loss()
            self.criterionRotFlow = torch.nn.L1Loss()
            
            # initialize optimizers            
            self.optimizer_G = torch.optim.Adam((self.netG_AB.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam((self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_AB)
        networks.print_network(self.netG_gc_AB)
        if self.isTrain:
            networks.print_network(self.netD_B)
            networks.print_network(self.netD_gc_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        
        self.gt_B = input['BtoA'] # The ground truth image of domain B
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def backward_D_basic(self, netD, real, fake, real2):
        loss_D_real, loss_D_distort = 0, 0
        '''
        Similar to ACGAN loss, but we train the classifier adversarially
        '''
        # Real_clean
        pred_real, pred_distort = self.forward_D_basic(netD, real)
        loss_D_real += self.criterionGAN(pred_real, True)
        loss_D_distort += self.criterionBCE(pred_distort, self.false)

        # Real_distort
        pred_real, pred_distort = self.forward_D_basic(netD, real2)
        # loss_D_real += self.criterionGAN(pred_real, True)
        loss_D_distort += self.criterionBCE(pred_distort, self.true)

        # Fake_clean
        pred_real, pred_distort = self.forward_D_basic(netD, fake)
        loss_D_real += self.criterionGAN(pred_real, False)
        # loss_D_distort += self.criterionBCE(pred_distort, self.false)
        loss_D_distort += self.criterionBCE(pred_distort, self.true) # adversarial

        loss_D = loss_D_real + loss_D_distort
        loss_D.backward()
        return loss_D_real, loss_D_distort

    def get_image_paths(self):
        return self.image_paths

    def rot(self, deg):
        '''
        Rotate image clockwisely by "deg" degree.
        Return callback function rather than value.
        '''
        def callback(tensor):
            size = self.opt.fineSize
            inv_idx = torch.arange(size-1, -1, -1).long().cuda()
            _iter = deg // 90
            for _ in range(_iter):
                # In each iterarion, the image rotates 90 deg
                tensor = tensor.transpose(2, 3)
                tensor = torch.index_select(tensor, 3, inv_idx)
            return tensor
        return callback

    def forward(self):
        input_A = self.input_A.clone()
        input_B = self.input_B.clone()

        self.real_A = self.input_A
        self.real_B = self.input_B

        size = self.opt.fineSize

        # Randomly choose a rotation degree from {90, 180, 270} and construct transfrom and inv transform fucntion
        if self.opt.no_rot:
            deg = 0
        else:
            deg = random.randint(1, 3) * 90
        
        self.tran = self.rot(deg)
        self.inv_tran = self.rot(360 - deg)

        self.real_gc_A = self.tran(input_A)
        self.real_gc_B = self.tran(input_B)
        
    def forward_G_basic(self, netG, real):
        '''
            Forward netG once.
            We downsample the real image, and then put it into the netG.
            After, we upsample the flow generated by netG back to the same size as the original real image.
        '''
        # downsampling
        real_down = F.interpolate(real, scale_factor=1 / self.opt.upsample_flow)
        flow = netG.forward(real_down)
        # upsampling
        flow = F.interpolate(flow, scale_factor=self.opt.upsample_flow).permute(0, 2, 3, 1)
        # construct offset grid
        self.theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).repeat(self.input_A.shape[0], 1, 1)
        self.grid = F.affine_grid(self.theta, self.input_A.shape).cuda() 
        # warp the real image by flow
        fake = F.grid_sample(real, flow + self.grid, padding_mode="zeros")
        return (fake, flow)
    
    def cal_smooth(self, flow):
        '''
            Smooth constraint for flow map.
        '''
        gx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])  # NCHW
        gy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])  # NCHW
        smooth = torch.mean(gx) + torch.mean(gy)
        return smooth
    
    def radial_constraint(self, flow):
        '''
            Radial constraint for flow map.
            The flow should point inward or outward to the center.
        '''
        n, h, w, _ = flow.shape
        # construct meshgrid
        x = torch.arange(0, h, dtype = torch.float).unsqueeze(1).repeat(1, w) - (h - 1) / 2
        y = torch.arange(0, w, dtype = torch.float).unsqueeze(0).repeat(h, 1) - (w - 1) / 2
        v = torch.cat((y.unsqueeze(2), x.unsqueeze(2)), dim = 2).unsqueeze(0).repeat(n, 1, 1, 1).cuda()
        
        v = v.view(n, h*w, 2)
        flow = flow.view(n, h*w, 2)
        # calcuate consine similarity of the flow and the mesh grid . 
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_similarity = torch.abs(cos(v, flow)) # May point to same or negative direction
        radial_loss = torch.mean(cos_similarity)
        return -radial_loss

    def circle_constraint(self, flow):
        '''
            Convert the map the polar axis, and then sample some radius and degree.
            For the points on the same circle, the should have the same length of flow.
            We need to minimize the variance of the length of flow on each circle.
        '''
        total_std = 0
        flow_len = (flow[:, :, :, 0] ** 2 + flow[:, :, :, 1] ** 2) ** (1 / 2)

        n, h, w, c = flow.shape
        size = h
        offset = size // 2
        n_sample = 256

        rad_list = np.random.uniform(0, (size // 2) * np.sqrt(2), n_sample)
        deg_list = np.random.uniform(0, 360, n_sample)
        
        for r in rad_list:
            X = (r * np.cos(deg_list) + offset).astype(np.int)
            Y = (r * np.sin(deg_list) + offset).astype(np.int)
            # ensure it is in the range
            indexs = (X >= 0) & (X < size) & (Y >= 0) & (Y < size)
            X = X[indexs]
            Y = Y[indexs]
            if len(X) > 1:
                total_std += torch.std(flow_len[:, X, Y])

        return total_std
    
    def forward_D_basic(self, netD, _input):
        return netD(_input)

    def backward_G(self):
        fake_B, flow_A = self.forward_G_basic(self.netG_AB, self.real_A)
        pred_real, pred_distort = self.forward_D_basic(self.netD_B, fake_B)
        loss_G_AB = ( self.criterionGAN(pred_real, True) + self.criterionBCE(pred_distort, self.false) )*self.opt.lambda_G

        fake_gc_B, flow_gc_A = self.forward_G_basic(self.netG_gc_AB, self.real_gc_A)
        pred_real, pred_distort = self.forward_D_basic(self.netD_gc_B, fake_gc_B)
        loss_G_gc_AB = ( self.criterionGAN(pred_real, True) + self.criterionBCE(pred_distort, self.false) )*self.opt.lambda_G

        # Constraints for flow map
        loss_crossflow = self.criterionCrossFlow(flow_A, flow_gc_A)*self.opt.lambda_crossflow
        loss_smooth = (self.cal_smooth(flow_A) + self.cal_smooth(flow_gc_A)) * self.opt.lambda_smooth
        loss_radialflow = (self.radial_constraint(flow_A) + self.radial_constraint(flow_gc_A)) * self.opt.lambda_radial
        loss_rotflow = (self.circle_constraint(flow_A) + self.circle_constraint(flow_gc_A)) * self.opt.lambda_rot
        
        # Geometry constraint
        loss_gc = self.get_gc_rot_loss(fake_B, fake_gc_B, 0) * self.opt.lambda_gc

        loss_G = loss_G_AB + loss_G_gc_AB + loss_gc + loss_crossflow + loss_radialflow + loss_smooth + loss_rotflow

        loss_G.backward()

        self.flow_A = flow_A
        self.flow_gc_A = flow_gc_A

        self.fake_B = fake_B.data
        self.fake_gc_B = fake_gc_B.data

        self.loss_G_AB = loss_G_AB.item()
        self.loss_G_gc_AB= loss_G_gc_AB.item()
        self.loss_gc = loss_gc.item()
        self.loss_crossflow = loss_crossflow.item()
        self.loss_radialflow = loss_radialflow.item()
        self.loss_smooth = loss_smooth.item()
        self.loss_rotflow = loss_rotflow.item()

    def get_gc_rot_loss(self, AB, AB_gc, direction):
        loss_gc = 0.0

        AB_gt = self.inv_tran(AB_gc.clone().detach())
        loss_gc = self.criterionGc(AB, AB_gt)
        AB_gc_gt = self.tran(AB.clone().detach())
        loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc*self.opt.lambda_AB*self.opt.lambda_gc
        return loss_gc

    def backward_D_B(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_gc_B = self.fake_gc_B_pool.query(self.fake_gc_B)
        loss_D_B_real, loss_D_B_distort = 0, 0
        tmp = self.backward_D_basic(self.netD_B, self.real_B, fake_B.detach(), self.real_A.detach())
        loss_D_B_real, loss_D_B_distort = loss_D_B_real + tmp[0], loss_D_B_distort + tmp[1]

        tmp = self.backward_D_basic(self.netD_gc_B, self.real_gc_B, fake_gc_B.detach(), self.real_gc_A.detach())
        loss_D_B_real, loss_D_B_distort = loss_D_B_real + tmp[0], loss_D_B_distort + tmp[1]
        
        self.loss_D_B_real = loss_D_B_real.item()
        self.loss_D_B_distort = loss_D_B_distort.item()
 
    def optimize_parameters(self):
        # forward
        self.forward()
        # G_AB
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_B and D_gc_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_B_real', self.loss_D_B_real), ('D_B_distort', self.loss_D_B_distort), ('G_AB', self.loss_G_AB),
                                  ('Gc', self.loss_gc), ('G_gc_AB', self.loss_G_gc_AB), ('Smooth', self.loss_smooth),
                                  ('Crossflow', self.loss_crossflow), ('Radial-flow', self.loss_radialflow), 
                                  ('Rotation-flow', self.loss_rotflow)])
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        gt_B = util.tensor2im(self.gt_B.data)
        fake_B = util.tensor2im(self.fake_B)

        chess_A = F.grid_sample(self.chess, (self.flow_A + self.grid)[0].unsqueeze(0))
        chess_A = util.tensor2im(chess_A.data)
        
        flow_map = plot_quiver(self.flow_A[0])
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('gt_B', gt_B), ('chess_A', chess_A), ('flow_map', flow_map)])
        self.ret_visuals = ret_visuals
        return ret_visuals
 
    def save(self, label):
        self.save_network(self.netG_AB, 'G_AB', label, self.gpu_ids)
        self.save_network(self.netG_gc_AB, 'G_AB', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.netD_gc_B, 'D_B', label, self.gpu_ids)

    def test(self):
        self.real_A = Variable(self.input_A)
        fake_B, flow_A = self.forward_G_basic(self.netG_AB, self.real_A)
        self.flow_A = flow_A
        self.fake_B = fake_B.data
