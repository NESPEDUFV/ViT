import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import Embedding 

# ---------- configuração ----------
IMAGE_PATH = "laranja.jpg" 
IMAGE_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM  = 64

# ---------- 1. abrir a imagem ----------
try:
    img = Image.open(IMAGE_PATH).convert("RGB")
except FileNotFoundError:
    print("Imagem não encontrada, usando uma imagem aleatória.")
    img = Image.fromarray(np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8))

# ---------- 2. imagem -> tensor (B, C, H, W) ----------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),          # (C, H, W), valores em [0, 1]
])
x = transform(img).unsqueeze(0)     # (1, 3, H, W)
print("Entrada:", tuple(x.shape))

# ---------- 3. rodar o embedding ----------
emb = Embedding(n_channels=3, image_size=IMAGE_SIZE,
                patch_size=PATCH_SIZE, embbed_dim=EMBED_DIM, dropout=0.0)
emb.eval()
out = emb(x)
n_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
patches = out[:, 1:, :]             # tira o [CLS], fica só com os patches

print("Saída completa:", tuple(out.shape),
      "| esperado:", (1, n_patches + 1, EMBED_DIM))
print("Só patches:    ", tuple(patches.shape),
      "| esperado:", (1, n_patches, EMBED_DIM))

# ---------- 4. verificações ----------
assert out.shape == (1, n_patches + 1, EMBED_DIM), "shape da saída errado"

# (a) determinismo (dropout = 0)
with torch.no_grad():
    assert torch.equal(out, emb(x)), "mesma imagem deu saídas diferentes"

with torch.no_grad():
    manual = emb.conv1(x[:, :, :PATCH_SIZE, :PATCH_SIZE]).flatten(1)   # (1, E)
    manual = manual + emb.pos_embedding[0, 0]
assert torch.allclose(out[0, 1], manual[0], atol=1e-6), \
    "patch 0 não bate com o recorte manual (confira reshape/permute)"

# (c) patches diferentes -> embeddings diferentes
assert not torch.allclose(patches[0, 0], patches[0, 1]), "patches idênticos?"

print("Tudo certo: os embeddings dos patches estão corretos.")