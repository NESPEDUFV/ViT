import os
import glob
import torch
from PIL import Image
from torchvision import datasets, transforms


class FMoWDataset(torch.utils.data.Dataset):
    """
    fMoW-RGB (jbourcier/fmow-rgb-baseline), já extraído em disco.

    Estrutura esperada dentro de <root>:
        <split>/<classe>/<classe>_<seq>/<classe>_<aoi>/<classe>_<seq>_<idx>_rgb.jpg
    As imagens já vêm recortadas e redimensionadas para 224x224 (8 bits, RGB).

    Parameters:
        root (str)         : pasta que contém as subpastas 'train' e 'val'
        split (str)        : 'train' ou 'val'
        transform          : transformações aplicadas à imagem (PIL -> tensor)
        class_to_idx (dict): mapa classe -> índice. Se None, é criado a partir das pastas
                             do split (ordem alfabética). O 'val' deve receber o mapa do 'train'.

    Returns (por item):
        (Tensor C,H,W ; int com o índice da classe)
    """
    def __init__(self, root, split, transform=None, class_to_idx=None):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Pasta '{split_dir}' não encontrada. "
                                    f"Extraia os .tar.gz (download.sh) e aponte --data_path para a pasta que contém '{split}/'.")

        if class_to_idx is None:
            classes = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))
            class_to_idx = {c: i for i, c in enumerate(classes)}

        self.class_to_idx = class_to_idx
        self.classes      = list(class_to_idx.keys())      # na ordem dos índices
        self.transform    = transform

        # <classe>/<classe>_<seq>/<classe>_<aoi>/<arquivo>_rgb.jpg
        paths = sorted(glob.glob(os.path.join(split_dir, '*', '*', '*', '*_rgb.jpg')))

        self.samples = []
        for p in paths:
            cls = os.path.relpath(p, split_dir).split(os.sep)[0]
            if cls in class_to_idx:
                self.samples.append((p, class_to_idx[cls]))

        if len(self.samples) == 0:
            raise RuntimeError(f"Nenhuma imagem '*_rgb.jpg' encontrada em '{split_dir}'. Confira a estrutura de pastas.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')               # garante 3 canais
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_loader(args):
    if args.dataset == 'mnist':
        # Transforms for train
        train_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                            transforms.RandomCrop(args.image_size, padding=2), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5], [0.5])])
        train = datasets.MNIST(os.path.join(args.data_path, args.dataset), train=True, download=True, transform=train_transform)

        # Transforms for test
        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize([0.5], [0.5])])
        test = datasets.MNIST(os.path.join(args.data_path, args.dataset), train=False, download=True, transform=test_transform)


    elif args.dataset == 'fashionmnist':
        train_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                            transforms.RandomCrop(args.image_size, padding=2), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5], [0.5])])
        train = datasets.FashionMNIST(os.path.join(args.data_path, args.dataset), train=True, download=True, transform=train_transform)

        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize([0.5], [0.5])])
        test = datasets.FashionMNIST(os.path.join(args.data_path, args.dataset), train=False, download=True, transform=test_transform)


    elif args.dataset == 'svhn':
        train_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                            transforms.RandomCrop(args.image_size, padding=2), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614])])
        train = datasets.SVHN(os.path.join(args.data_path, args.dataset), split='train', download=True, transform=train_transform)

        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize([0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614])])
        test = datasets.SVHN(os.path.join(args.data_path, args.dataset), split='test', download=True, transform=test_transform)

    elif args.dataset == 'cifar10':
        train_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                            transforms.RandomCrop(args.image_size, padding=4), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandAugment(),  # RandAugment augmentation for strong regularization
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        train = datasets.CIFAR10(os.path.join(args.data_path, args.dataset), train=True, download=True, transform=train_transform)

        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        test = datasets.CIFAR10(os.path.join(args.data_path, args.dataset), train=False, download=True, transform=test_transform)

    elif args.dataset == 'cifar100':
        train_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                            transforms.RandomCrop(args.image_size, padding=4), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandAugment(),  # RandAugment augmentation for strong regularization
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])])
        train = datasets.CIFAR100(os.path.join(args.data_path, args.dataset), train=True, download=True, transform=train_transform)

        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])])
        test = datasets.CIFAR100(os.path.join(args.data_path, args.dataset), train=False, download=True, transform=test_transform)

    elif args.dataset == 'fmow':
        # fMoW-RGB: as imagens já são 224x224. Média/desvio abaixo são os do ImageNet (provisórios).
        # Ideal: calcular a média/desvio do conjunto de treino do fMoW e substituir aqui.
        fmow_mean, fmow_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        # Imagens de satélite não têm "lado de cima" fixo, então flips verticais fazem sentido
        train_transform = transforms.Compose([transforms.RandomResizedCrop(args.image_size, scale=(0.6, 1.0)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(fmow_mean, fmow_std)])
        train = FMoWDataset(args.data_path, 'train', transform=train_transform)

        # Padrão do fMoW: 'val' serve como validação e teste. Usa o mesmo mapa de classes do treino.
        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                             transforms.ToTensor(),
                                             transforms.Normalize(fmow_mean, fmow_std)])
        test = FMoWDataset(args.data_path, 'val', transform=test_transform, class_to_idx=train.class_to_idx)

        args.n_classes = len(train.classes)                 # o número de classes vem dos dados
        print(f"fMoW: {len(train)} imagens de treino, {len(test)} de validação, {args.n_classes} classes")

    else:
        print("Unknown dataset")
        exit(0)

    # Define dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.n_workers,
                                               pin_memory=args.is_cuda,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=args.batch_size*2,
                                              shuffle=False,
                                              num_workers=args.n_workers,
                                              pin_memory=args.is_cuda,
                                              drop_last=False)

    return train_loader, test_loader
