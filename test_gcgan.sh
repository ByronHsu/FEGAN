# free
python3 test.py --dataroot ./datasets/MCindoor_fisheye --name tmp --model gc_gan_cross --no_dropout --which_model_netG unet_128 --batchSize 1 --which_direction BtoA --geometry rot --which_epoch 60
# radial
python3 test.py --dataroot ./datasets/MCindoor_fisheye --name 10_01_0044 --model gc_gan_cross --no_dropout --which_model_netG unet_128 --batchSize 1 --which_direction BtoA --geometry rot --which_epoch 60