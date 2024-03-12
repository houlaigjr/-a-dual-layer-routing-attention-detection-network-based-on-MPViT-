To train MPViT models on ImageNet on a single node with 8 gpus for 300 epochs, run:

python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use>  main.py \ 
--model <mpvit_model> --data-path <imagenet-path> --batch-size <batch-size-per-gpu> --output <output-directory>
MPViT-Tiny:

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--model mpvit_tiny --batch-size 128 --data-path <imagenet-path> --output_dir <output-directory>
MPViT-Xsmall:

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--model mpvit_xsmall --batch-size 128 --data-path <imagenet-path> --output_dir <output-directory>
MPViT-Small:

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--model mpvit_small --batch-size 128 --drop-path 0.05 --data-path <imagenet-path> --output_dir <output-directory>
MPViT-Base:

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--model mpvit_base --batch-size 128 --drop-path 0.3 --data-path <imagenet-path> --output_dir <output-directory>
