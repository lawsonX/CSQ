## CS-BSQ 


1. Change Imagenet train set and val set to lmdb

```
python folder2lmdb.py
```

2. train Cifar10 
    
    Only support Single GPU training

```
CUDA_VISIBLE_DEVICES=0 python main_cifar.py --classes 10 
```
3. train Tiny-Imagenet
    
    Only support Single GPU training

```
CUDA_VISIBLE_DEVICES=0 python main_cifar.py --classes 200 
```

4. Train Imagenet

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 distributed_imagenet.py
```