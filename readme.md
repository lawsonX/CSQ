## CSQ 
    
Implementation for Growing Mixed-Precision Quantization Scheme with Bi-level Continuous Sparsification (CSQ)
 
### core of bit-level continuous sparsification
    bits/bitcs.py BitLinear, BitConv2d
    

### Run experiments
1. train Cifar10 
    
    Only support Single GPU training, no need to train rounds

```
CUDA_VISIBLE_DEVICES=0 python main_cifar.py \
--classes 10 --arch "ResNet" \
--batch-size 256 --lr 0.1 --warmup --epochs 600 --ticket 595 \
--Nbits 8 --target 2 --act 0 --final-temp 200 --t0 1 --lmbda 0.01 \
--save_file "cifar_resnet20_n8t2a0" 
```

2. Train Imagenet-1000

    supporting ddp; need to train a few rounds

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 dist_img_cs.py \
--data "/home/zhen/imagenet12" \
--classes 1000 --arch "ResNet50" \
--batch-size 512 --lr 0.1 --warmup --t0 1 \
--rounds 3 --rewind 2 --lmbda 0.001 \
--epochs 90 \
--Nbits 8 --target 4 --act 4 \
--save_file "IMG_res50_n8t4a4"
```

### Related Codebase

https://github.com/yanghr/BSQ

https://github.com/lolemacs/continuous-sparsification

