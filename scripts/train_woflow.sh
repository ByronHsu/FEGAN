if [[ $# -lt 2 ]] ; then
    echo 'bash train_gcgan.sh [dataset_path] [name]'
    exit 0
fi
python3 \
train.py \
--dataroot $1 \
--name $2 \
--model gc_gan_cross \
--batchSize 4 \
--niter 80 \
--niter_decay 20 \
--save_epoch_freq 2 \
--which_direction BtoA \
--tensorboard \
--nThreads 0 \
--which_model_netG unet_128 \
--upsample_flow 2 \
--use_att \
--geometry rot \
--identity 0 \
--GD_share \
--which_model_netD Fusion \
--lambda_gc 1 \
--lambda_smooth 0 \
--lambda_crossflow 0 \
--lambda_radial 0 \
--lambda_rot 0 \