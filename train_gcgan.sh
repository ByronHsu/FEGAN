
python3 train.py --dataroot ./datasets/fisheye_gcgan --name 10_01_0812 --model gc_gan_cross --batchSize 4 --display_id 1 --which_direction BtoA --which_model_netG unet_128 --which_model_netD n_layers --n_layers_D 4 --geometry rot --nThreads 0 --lambda_smooth 1 --lambda_crossflow 1 --lambda_selfflow 0.1 --lambda_rot 0 --no_dropout --identity 0
