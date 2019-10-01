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
from PIL import Image

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
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_gc_AB = networks.define_G(opt.input_nc, flow_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        # self.netG_gc_AB = self.netG_AB # share

        self.true = True # torch.ones((nb, 1)).cuda()
        self.false = False # torch.zeros((nb, 1)).cuda()

        
        self.offset = torch.zeros(nb, size, size, 2).cuda() #
        
        
        img_path = os.path.join(os.getcwd(), 'chessboard.jpg')
        img = Image.open(img_path)
        
        transform1 = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor() # range [0, 255] -> [0.0,1.0]
            ]
        )
        self.chess = transform1(img).unsqueeze(0).cuda()

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_gc_B = networks.define_D(opt.output_nc, opt.ndf,
                                               opt.which_model_netD,
                                               opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_AB, 'G_AB', which_epoch)
            self.load_network(self.netG_gc_AB, 'G_gc_AB', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_B, 'D_B', which_epoch)
                self.load_network(self.netD_gc_B, 'D_gc_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_gc_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor) # torch.nn.BCEWithLogitsLoss() #
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionGc = torch.nn.L1Loss()
            self.criterionCrossFlow = torch.nn.L1Loss() # constraint 1: two generated flow should be the same due to symmetry
            self.criterionRotFlow = torch.nn.L1Loss()
            # initialize optimizers
            
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AB.parameters(), self.netG_gc_AB.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            # share
            # self.optimizer_G = torch.optim.Adam(self.netG_AB.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(itertools.chain(self.netD_B.parameters(), self.netD_gc_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

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
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def backward_D_basic(self, netD, real, fake, netD_gc, real_gc, fake_gc):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, self.true)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, self.false)
        # Fake2
        pred_fake2 = netD(self.input_A.detach())
        loss_D_fake2 = self.criterionGAN(pred_fake2, self.false)

        # Combined loss
        loss_D = (loss_D_real + loss_D_fake2) + loss_D_fake

        # Real_gc
        pred_real_gc = netD_gc(real_gc)
        loss_D_gc_real = self.criterionGAN(pred_real_gc, self.true)
        # Fake_gc
        pred_fake_gc = netD_gc(fake_gc.detach())
        loss_D_gc_fake = self.criterionGAN(pred_fake_gc, self.false)
        # Fake_gc2
        pred_fake_gc2 = netD_gc(self.real_gc_A.detach())
        loss_D_gc_fake2 = self.criterionGAN(pred_fake_gc2, self.false)
        # print(pred_real_gc.shape)
        # Combined loss
        loss_D += ((loss_D_gc_real + loss_D_gc_fake2) + loss_D_gc_fake)

        # backward
        loss_D.backward()
        return loss_D

    def get_image_paths(self):
        return self.image_paths

    def rot90(self, tensor, direction):
        tensor = tensor.transpose(2, 3)
        size = self.opt.fineSize
        inv_idx = torch.arange(size-1, -1, -1).long().cuda()
        if direction == 0:
          tensor = torch.index_select(tensor, 3, inv_idx)
        else:
          tensor = torch.index_select(tensor, 2, inv_idx)
        return tensor

    def forward(self):
        input_A = self.input_A.clone()
        input_B = self.input_B.clone()

        self.real_A = self.input_A
        self.real_B = self.input_B

        size = self.opt.fineSize

        if self.opt.geometry == 'rot':
          self.real_gc_A = self.rot90(input_A, 0)
          self.real_gc_B = self.rot90(input_B, 0)
        elif self.opt.geometry == 'vf':
          inv_idx = torch.arange(size-1, -1, -1).long().cuda()
          self.real_gc_A = torch.index_select(input_A, 2, inv_idx)
          self.real_gc_B = torch.index_select(input_B, 2, inv_idx)
        else:
          raise ValueError("Geometry transformation function [%s] not recognized." % opt.geometry)
        
    def forward_G_basic(self, netG, real):
        real_down = F.interpolate(real, scale_factor=0.5)
        # print(real_down.shape)
        flow = netG.forward(real_down)
        # flow = netG.forward(real).permute(0, 2, 3, 1)
        # print('flow', flow.shape)
        # input()
        flow = F.upsample(flow, scale_factor=2).permute(0, 2, 3, 1)
        # print(flow.shape, real_down.shape)
        self.theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).repeat(self.input_A.shape[0], 1, 1)
        self.grid = F.affine_grid(self.theta, self.input_A.shape).cuda() 
        fake = F.grid_sample(real, flow + self.grid, padding_mode="zeros")
        return (fake, flow)
    
    def cal_smooth(self, flow):
        '''
            Smooth constraint for flow map
        '''
        gx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])  # NCHW
        gy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])  # NCHW
        smooth = torch.mean(gx) + torch.mean(gy)
        return smooth
    
    def radial_constraint(self, flow):
        '''
            Flow map has dimension (n, h, w, 2).
        '''
        n, h, w, _ = flow.shape
        x = torch.arange(0, h, dtype = torch.float).unsqueeze(1).repeat(1, w) - (h - 1) / 2
        y = torch.arange(0, w, dtype = torch.float).unsqueeze(0).repeat(h, 1) - (w - 1) / 2
        v = torch.cat((y.unsqueeze(2), x.unsqueeze(2)), dim = 2).unsqueeze(0).repeat(n, 1, 1, 1).cuda()
        
        v = v.view(n, h*w, 2)
        flow = flow.view(n, h*w, 2)
        
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_similarity = torch.abs(torch.abs(cos(v, flow)) - 1)
        radial_loss = torch.mean(cos_similarity)
        return radial_loss

    def rotation_constraint(self, flow):
        flow = flow.permute(0, 3, 1, 2)
        rot90_flow = self.rot90(flow, 0)
        rot180_flow = self.rot90(rot90_flow, 0)
        rot270_flow = self.rot90(rot180_flow, 0)

        criterion = self.criterionRotFlow
        rot_loss = (criterion(flow, rot90_flow) + criterion(flow, rot180_flow) + criterion(flow, rot270_flow))
        return rot_loss

    def selfFlowLoss(self, flow):
        return self.radial_constraint(flow)
    
    def backward_G(self):
        
        fake_B, flow_A = self.forward_G_basic(self.netG_AB, self.real_A)
        pred_fake = self.netD_B.forward(fake_B)
        loss_G_AB = self.criterionGAN(pred_fake, self.true)*self.opt.lambda_G

        fake_gc_B, flow_gc_A = self.forward_G_basic(self.netG_gc_AB, self.real_gc_A)
        pred_fake = self.netD_gc_B.forward(fake_gc_B)
        loss_G_gc_AB = self.criterionGAN(pred_fake, self.true)*self.opt.lambda_G

        # Constraints for flow map
        loss_crossflow = self.criterionCrossFlow(flow_A, flow_gc_A)*self.opt.lambda_crossflow
        loss_smooth = (self.cal_smooth(flow_A) + self.cal_smooth(flow_gc_A)) * self.opt.lambda_smooth
        loss_selfflow = (self.selfFlowLoss(flow_A) + self.selfFlowLoss(flow_gc_A)) * self.opt.lambda_selfflow
        loss_rotflow = (self.rotation_constraint(flow_A) + self.rotation_constraint(flow_gc_A)) * self.opt.lambda_rot
        if self.opt.geometry == 'rot':
            loss_gc = self.get_gc_rot_loss(fake_B, fake_gc_B, 0)
        elif self.opt.geometry == 'vf':
            loss_gc = self.get_gc_vf_loss(fake_B, fake_gc_B)

        if self.opt.identity > 0:
            # G_AB should be identity if real_B is fed.
            idt_A, flow = self.forward_G_basic(self.netG_AB, self.real_B)
            loss_idt = self.criterionIdt(idt_A, self.real_B) * self.opt.lambda_AB * self.opt.identity
            #loss_s += self.cal_smooth(flow)

            idt_gc_A, flow = self.forward_G_basic(self.netG_gc_AB, self.real_gc_B)
            loss_idt_gc = self.criterionIdt(idt_gc_A, self.real_gc_B) * self.opt.lambda_AB * self.opt.identity
            #loss_s += self.cal_smooth(flow)

            self.idt_A = idt_A.data
            self.idt_gc_A = idt_gc_A.data
            self.loss_idt = loss_idt.item()
            self.loss_idt_gc = loss_idt_gc.item()
        else:
            loss_idt = 0
            loss_idt_gc = 0
            self.loss_idt = 0
            self.loss_idt_gc = 0

        loss_G = loss_G_AB + loss_G_gc_AB + loss_gc + loss_idt + loss_idt_gc + loss_crossflow + loss_selfflow + loss_smooth

        loss_G.backward()

        self.flow_A = flow_A
        self.flow_gc_A = flow_gc_A

        self.fake_B = fake_B.data
        self.fake_gc_B = fake_gc_B.data

        self.loss_G_AB = loss_G_AB.item()
        self.loss_G_gc_AB= loss_G_gc_AB.item()
        self.loss_gc = loss_gc.item()
        self.loss_crossflow = loss_crossflow.item()
        self.loss_selfflow = loss_selfflow.item()
        self.loss_smooth = loss_smooth.item()
        self.loss_rotflow = loss_rotflow.item()

    def get_gc_rot_loss(self, AB, AB_gc, direction):
        loss_gc = 0.0

        if direction == 0:
          AB_gt = self.rot90(AB_gc.clone().detach(), 1)
          loss_gc = self.criterionGc(AB, AB_gt)
          AB_gc_gt = self.rot90(AB.clone().detach(), 0)
          loss_gc += self.criterionGc(AB_gc, AB_gc_gt)
        else:
          AB_gt = self.rot90(AB_gc.clone().detach(), 0)
          loss_gc = self.criterionGc(AB, AB_gt)
          AB_gc_gt = self.rot90(AB.clone().detach(), 1)
          loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc*self.opt.lambda_AB*self.opt.lambda_gc
        return loss_gc

    def get_gc_vf_loss(self, AB, AB_gc):
        loss_gc = 0.0

        size = self.opt.fineSize

        inv_idx = torch.arange(size-1, -1, -1).long().cuda()

        AB_gt = torch.index_select(AB_gc.clone().detach(), 2, inv_idx)
        loss_gc = self.criterionGc(AB, AB_gt)

        AB_gc_gt = torch.index_select(AB.clone().detach(), 2, inv_idx)
        loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc*self.opt.lambda_AB*self.opt.lambda_gc
        return loss_gc

    def backward_D_B(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_gc_B = self.fake_gc_B_pool.query(self.fake_gc_B)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B, self.netD_gc_B, self.real_gc_B, fake_gc_B)
        self.loss_D_B = loss_D_B.item()
 
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
        ret_errors = OrderedDict([('D_B', self.loss_D_B), ('G_AB', self.loss_G_AB),
                                  ('Gc', self.loss_gc), ('G_gc_AB', self.loss_G_gc_AB), ('Smooth', self.loss_smooth),
                                  ('Crossflow', self.loss_crossflow), ('Self-flow', self.loss_selfflow), 
                                  ('Rotation-flow', self.loss_rotflow)])

        if self.opt.identity > 0.0:
            ret_errors['idt'] = self.loss_idt
            ret_errors['idt_gc'] = self.loss_idt_gc

        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        real_B = util.tensor2im(self.real_B.data)

        fake_B = util.tensor2im(self.fake_B)
        fake_gc_B = util.tensor2im(self.fake_gc_B)

        
        chess_A = F.grid_sample(self.chess, (self.flow_A + self.grid)[0].unsqueeze(0))
        chess_A = util.tensor2im(chess_A.data)
        
        flow_map = plot_quiver(self.flow_A[0])# use clamp to avoid too large/small value ruins the relative scale
        # print(self.flow_A[0] + self.grid[0])

        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake), ('chess_A', chess_A), ('flow_map', flow_map)])
         

        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_AB, 'G_AB', label, self.gpu_ids)
        self.save_network(self.netG_gc_AB, 'G_gc_AB', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.netD_gc_B, 'D_gc_B', label, self.gpu_ids)

    def test(self):
        self.real_A = Variable(self.input_A)
        self.real_gc_A = self.rot90(self.input_A, 0)
        self.real_B = Variable(self.input_B)
        
        fake_B, flow_A = self.forward_G_basic(self.netG_AB, self.real_A)
        fake_gc_B, flow_gc_A = self.forward_G_basic(self.netG_gc_AB, self.real_gc_A)
        self.flow_A = flow_A
        self.fake_B = fake_B.data
        self.fake_gc_B = fake_gc_B.data
