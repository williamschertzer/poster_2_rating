import torch
from torch import nn

class PosterEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ParameterList([ # 3 * 256 * 256
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.Conv2d(3, 8, 6, 2), # 8 * 126 * 126
            nn.Conv2d(8, 8, 5, 1, 2),
            nn.Conv2d(8, 8, 5, 1, 2),
            nn.Conv2d(8, 32, 6, 2), # 32 * 61 * 61
            nn.Conv2d(32, 32, 7, 1, 3),
            nn.Conv2d(32, 32, 7, 1, 3),
            nn.Conv2d(32, 128, 7, 2), # 128 * 28 * 28
            nn.Conv2d(128, 128, 9, 1, 4),
            nn.Conv2d(128, 128, 9, 1, 4),
            nn.Conv2d(128, 512, 7, 3), # 512 * 8 * 8
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
        ])
        self.lrelu = nn.LeakyReLU()
    def forward(self, input: torch.Tensor):
        """
        input: Batch x 3 x 256 x 256
        output: Batch x (512 * 8 * 8)
        """
        for conv in self.convs:
            input = self.lrelu(conv(input))
        return torch.flatten(input, -3) # batch * (512 * 8 * 8)

class PosterDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # inp: 512 * 8 * 8
        self.convs = nn.ParameterList([
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ConvTranspose2d(512, 128, 7, 3), # 128 * 28 * 28
            nn.Conv2d(128, 128, 9, 1, 4),
            nn.Conv2d(128, 128, 9, 1, 4),
            nn.ConvTranspose2d(128, 32, 7, 2), # 32 * 61 * 61
            nn.Conv2d(32, 32, 7, 1, 3),
            nn.Conv2d(32, 32, 7, 1, 3),
            nn.ConvTranspose2d(32, 8, 6, 2), # 8 * 126 * 126
            nn.Conv2d(8, 8, 5, 1, 2),
            nn.Conv2d(8, 8, 5, 1, 2),
            nn.ConvTranspose2d(8, 3, 6, 2), # 3 * 256 * 256
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.Conv2d(3, 3, 3, 1, 1),
        ])
        self.relu = nn.ReLU()
    def forward(self, input: torch.Tensor):
        """
        input: Batch x (512 * 8 * 8)
        output: Batch x 3 x 256 x 256
        """
        input = torch.unflatten(input, -1, (512, 8, 8))
        for layer in self.convs[:-1]:
            input = layer(input)
        output = (self.convs[-1](input))
        return output
    
class PosterAutoEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, input: torch.Tensor):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded

class FlatVGG(nn.Module):
    def __init__(self, vgg_model) -> None:
        super().__init__()
        self.vgg_model = vgg_model
    def forward(self, input):
        return torch.flatten(self.vgg_model(input), -3)

