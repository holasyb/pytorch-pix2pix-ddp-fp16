set -e
set -u

export CUDA_VISIBLE_DEVICES=0,1
gpunum=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

dataset=maps # maps just for test
dataroot=./datasets/$dataset
model_name=${dataset}_pix2pixgan
checkpoints_dir=./temp_checkpoints

# ddp only
# python -m torch.distributed.launch --nproc_per_node=$gpunum --master_port 12453 train.py --dataroot $dataroot --name $model_name --model pix2pix --checkpoints_dir $checkpoints_dir \
#     --direction BtoA --netG unet_256 --lambda_L1 100 --norm batch --pool_size 0 \
#     --input_nc 3 --batch_size 16 --preprocess none --num_threads 8 --n_epochs 150 \
#     --n_epochs_decay 0 --load_size 256 --crop_size 256 --lr_policy cosine \
#     --display_id -1 --no_html --lr 0.0002 --use_ddp --preprocess resize_and_crop

# ddp + fp16
# python -m torch.distributed.launch --nproc_per_node=$gpunum --master_port 12453 train.py --dataroot $dataroot --name $model_name --model pix2pix --checkpoints_dir $checkpoints_dir \
#     --direction BtoA --netG unet_256 --lambda_L1 100 --norm batch --pool_size 0 \
#     --input_nc 3 --batch_size 16 --preprocess none --num_threads 8 --n_epochs 150 \
#     --n_epochs_decay 0 --load_size 256 --crop_size 256 --lr_policy cosine \
#     --display_id -1 --no_html --lr 0.0002 --use_ddp --preprocess resize_and_crop --use_fp16

# no ddp, no fp16
# python train.py --dataroot $dataroot --name $model_name --model pix2pix --checkpoints_dir $checkpoints_dir \
#     --direction BtoA --netG unet_256 --lambda_L1 100 --norm batch --pool_size 0 \
#     --input_nc 3 --batch_size 16 --preprocess none --num_threads 8 --n_epochs 150 \
#     --n_epochs_decay 0 --load_size 256 --crop_size 256 --lr_policy cosine \
#     --display_id -1 --no_html --lr 0.0002 --preprocess resize_and_crop

# python train.py --dataroot $dataroot --name $model_name --model pix2pix --checkpoints_dir $checkpoints_dir \
#     --direction BtoA --netG unet_256 --lambda_L1 100 --norm batch --pool_size 0 \
#     --input_nc 3 --batch_size 16 --preprocess none --num_threads 8 --n_epochs 150 \
#     --n_epochs_decay 0 --load_size 256 --crop_size 256 --lr_policy cosine \
#     --display_id -1 --no_html --lr 0.0002 --preprocess resize_and_crop --use_fp16
