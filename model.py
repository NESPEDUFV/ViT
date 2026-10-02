import torch 
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, n_channels, image_size, patch_size, embbed_dim, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, embbed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size//patch_size) ** 2, embbed_dim), requires_grad=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embbed_dim), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(x)
        x = x.reshape([B, x.shape[1], -1])
        x = x.permute(0, 2, 1)
        x = x + self.pos_embedding
        x = torch.cat((torch.repeat_interleave(self.cls_token, B, 0), x), dim=1)
        x = self.dropout(x)
        return x


