# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 distributed_imagenet.py \
# --data "/home/zhen/imagenet12" \
# --classes 1000 --arch "ResNet18" \
# --batch-size 1024 --lr 0.1 \
# --epochs 300 --ticket 200 \
# --Nbits 6 --target 3 --act 0 \
# --save_file "IMG_res18_n6t3a0_lr1wp_e300t200"

CUDA_VISIBLE_DEVICES=6,7,8,9 python3 -m torch.distributed.launch --nproc_per_node=4 distributed_imagenet_cs.py \
--data "/home/zhen/imagenet12" \
--classes 1000 --arch "ResNet50" \
--batch-size 1024 --lr 0.1 --warmup --t0 1 \
--rounds 3 --rewind 2 --lmbda 0.1 \
--epochs 90 \
--Nbits 8 --target 4 --act 4 \
--save_file "IMG_res50_n8t4a4_e90r3"
