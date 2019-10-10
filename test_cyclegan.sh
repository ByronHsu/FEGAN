if [[ $# -lt 3 ]] ; then
    echo 'bash test_cyclegan.sh [dataset_path] [name] [which_epoch]'
    exit 0
fi
python3 test.py \
 --dataroot $1 \
 --name $2 \
 --model cycle_gan \
 --no_dropout \
 --which_model_netG resnet_6blocks \
 --batchSize 1 \
 --which_epoch $3 \
 --loadSize 256 \
 --fineSize 256 \
 --use_att