python main.py --dataset mnist --epochs 100
python main.py --dataset fashionmnist
python main.py --dataset svhn --n_channels 3 --image_size 32 --embed_dim 128
python main.py --dataset cifar10 --n_channels 3 --image_size 32 --embed_dim 128
python main.py --dataset cifar100 --n_channels 3 --image_size 32 --embed_dim 128 --n_classes 100
python main.py --dataset fmow --data_path /caminho/para/fmow-rgb-baseline --n_channels 3 --image_size 224 --patch_size 16 --embed_dim 192 --n_attention_heads 3 --n_layers 6 --forward_mul 4 --batch_size 128 --epochs 30 --warmup_epochs 5
