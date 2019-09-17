import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import random
import math
import sys
import pdb
import torch.nn.functional as F
import torchvision.transforms as transforms 
from PIL import Image

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
        self.netG_gc_AB = networks.define_G(opt.input_nc, flow_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        
        theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).repeat(4, 1, 1)
        self.grid = F.affine_grid(theta, self.input_A.shape).cuda()
        
        
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
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionGc = torch.nn.L1Loss()
            self.criterionCrossFlow = torch.nn.L1Loss() # constraint 1: two generated flow should be the same due to symmetry
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AB.parameters(), self.netG_gc_AB.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
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
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        # Real_gc
        pred_real_gc = netD_gc(real_gc)
        loss_D_gc_real = self.criterionGAN(pred_real_gc, True)
        # Fake_gc
        pred_fake_gc = netD_gc(fake_gc.detach())
        loss_D_gc_fake = self.criterionGAN(pred_fake_gc, False)
        # Combined loss
        loss_D += (loss_D_gc_real + loss_D_gc_fake) * 0.5

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

    def backward_G(self):
        # adversariasl loss
        flow_A = self.netG_AB.forward(self.real_A).permute(0, 2, 3, 1) # predicted flow map of A
        # print(flow_A + self.grid)
        fake_B = F.grid_sample(self.real_A, flow_A + self.grid, padding_mode = 'border')
        pred_fake = self.netD_B.forward(fake_B)
        loss_G_AB = self.criterionGAN(pred_fake, True)*self.opt.lambda_G

        flow_gc_A = self.netG_gc_AB.forward(self.real_gc_A).permute(0, 2, 3, 1) # predicted flow map of gc_A
        fake_gc_B = F.grid_sample(self.real_gc_A, flow_gc_A + self.grid, padding_mode = 'border')
        pred_fake = self.netD_gc_B.forward(fake_gc_B)
        loss_G_gc_AB = self.criterionGAN(pred_fake, True)*self.opt.lambda_G

        lambda_CrossFlow = 10
        loss_CrossFlow = self.criterionCrossFlow(flow_A, flow_gc_A)*lambda_CrossFlow

        if self.opt.geometry == 'rot':
            loss_gc = self.get_gc_rot_loss(fake_B, fake_gc_B, 0)
        elif self.opt.geometry == 'vf':
            loss_gc = self.get_gc_vf_loss(fake_B, fake_gc_B)

        if self.opt.identity > 0:
            # G_AB should be identity if real_B is fed.
            flow = self.netG_AB(self.real_B).permute(0, 2, 3, 1)
            idt_A = F.grid_sample(self.real_B, flow + self.grid, padding_mode = 'border')
            loss_idt = self.criterionIdt(idt_A, self.real_B) * self.opt.lambda_AB * self.opt.identity
            
            flow = self.netG_gc_AB(self.real_gc_B).permute(0, 2, 3, 1)
            idt_gc_A = F.grid_sample(self.real_gc_B, flow + self.grid, padding_mode = 'border')
            loss_idt_gc = self.criterionIdt(idt_gc_A, self.real_gc_B) * self.opt.lambda_AB * self.opt.identity

            self.idt_A = idt_A.data
            self.idt_gc_A = idt_gc_A.data
            self.loss_idt = loss_idt.item()
            self.loss_idt_gc = loss_idt_gc.item()
        else:
            loss_idt = 0
            loss_idt_gc = 0
            self.loss_idt = 0
            self.loss_idt_gc = 0

        loss_G = loss_G_AB + loss_G_gc_AB + loss_gc + loss_idt + loss_idt_gc + loss_CrossFlow

        loss_G.backward()

        self.flow_A = flow_A
        self.flow_gc_A = flow_gc_A

        self.fake_B = fake_B.data
        self.fake_gc_B = fake_gc_B.data

        self.loss_G_AB = loss_G_AB.item()
        self.loss_G_gc_AB= loss_G_gc_AB.item()
        self.loss_gc = loss_gc.item()
        self.loss_CrossFlow = loss_CrossFlow.item()

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
                                  ('Gc', self.loss_gc), ('G_gc_AB', self.loss_G_gc_AB), ('G_CF', self.loss_CrossFlow)])

        if self.opt.identity > 0.0:
            ret_errors['idt'] = self.loss_idt
            ret_errors['idt_gc'] = self.loss_idt_gc

        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        real_B = util.tensor2im(self.real_B.data)

        fake_B = util.tensor2im(self.fake_B)
        fake_gc_B = util.tensor2im(self.fake_gc_B)

        
        chess_A = F.grid_sample(self.chess, (self.grid + self.flow_A)[0].unsqueeze(0))
        chess_A = util.tensor2im(chess_A.data)

        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B), ('fake_gc_B', fake_gc_B), ('chess_A', chess_A)])

        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_AB, 'G_AB', label, self.gpu_ids)
        self.save_network(self.netG_AB, 'G_gc_AB', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.netD_gc_B, 'D_gc_B', label, self.gpu_ids)

    def test(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        
        flow_A = self.netG_AB.forward(self.real_A).data
        self.fake_B = F.grid_sample(self.real_A, flow_A + self.grid, padding_mode = 'border')