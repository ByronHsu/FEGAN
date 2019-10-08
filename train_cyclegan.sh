if [[ $# -lt 2 ]] ; then
    echo 'bash train_cyclegan.sh [dataset_path] [name]'
    exit 0
fi
python3 \
train.py --dataroot $1 \
--name $2 \
--model cycle_gan \
--pool_size 50 \
--no_dropout \
--batchSize 4 \
--tensorboard \
--niter 80 \
--niter_decay 20 \
--save_epoch_freq 3 \
--which_model_netG resnet_6blocks