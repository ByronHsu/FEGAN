import argparse
import os
#from solver import Solver
from torch.backends import cudnn
from data_loader import get_loader
import pdb

def str2bool(v):
    return v.lower() in ('true')

def main(config):

    svhn_loader, mnist_loader, svhn_test_loader, mnist_test_loader = get_loader(config)

    if config.geometry == 0:
        from solver import Solver
    elif config.geometry == 1:
        from solver_share_rot import Solver
    else:
        from solver_share_vf import Solver

    # train loader
    solver = Solver(config, svhn_loader, mnist_loader)
    cudnn.benchmark = True

    # create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)
    
    if config.mode == 'train':
        # test_loader here is just for test.
        solver.train(svhn_test_loader, mnist_test_loader)
    elif config.mode == 'test':
        # test_loader here is just for test.
        solver.test(svhn_test_loader, mnist_test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--use_reconst_loss', required=True, type=str2bool)
    parser.add_argument('--use_distance_loss', required=False, type=str2bool)
    parser.add_argument('--num_classes', type=int, default=10)
    
    # training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=40000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lambda_distance_A', type=float, default=0.05)
    parser.add_argument('--lambda_distance_B', type=float, default=0.1)
    parser.add_argument('--use_self_distance', required=False, type=str2bool)
    parser.add_argument('--max_items', type=int, default=400)
    parser.add_argument('--unnormalized_distances', required=False, type=str2bool)
    parser.add_argument('--lambda_gc', type=float, default=2.0)
    # 0:distanceGAN, 1:rot, 2:vf
    parser.add_argument('--geometry', type=int, default=0)
    
    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--mnist_path', type=str, default='./mnist')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--svhn_path', type=str, default='./svhn')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_step', type=int , default=500)

    config = parser.parse_args()
    print(config)
    main(config)
