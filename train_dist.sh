CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 distributed_imagenet.py \
--data "/home/zhen/imagenet12" \
--classes 100 --arch "ResNet18" \
--batch-size 1024 --lr 0.1 \
--epochs 300 --ticket 200 \
--Nbits 6 --target 3 --act 0 \
--save_file "IMG_res18_n6t3a0_lr1wp_e300t200"
