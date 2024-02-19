import torch
import torch.nn as nn

###############
# simple blocks
###############

def conv_block(in_features: int, out_features: int, kernel=(3, 3), stride=(1, 1), padding=(0, 0), activation='PReLU'):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel, stride, padding),
        getattr(nn, activation)(),
        nn.BatchNorm2d(out_features),
    )

def conv_transpose_block(in_features: int, out_features: int, kernel=(3, 3), stride=(1, 1), output_padding=(0, 0), activation='PReLU'):
    return nn.Sequential(
        getattr(nn, activation)(),
        nn.BatchNorm2d(in_features),
        nn.ConvTranspose2d(in_features, out_features, kernel, stride=stride, output_padding=output_padding),
    )

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(OutConv, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=7, activation="Sigmoid"):
        super().__init__()
        padding = int((kernel_size - 1) / 2)

        self.conv = nn.Sequential( # for old versions: double_conv
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            getattr(nn, activation)(),
            
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            getattr(nn, activation)(),
        )

    def forward(self, x):
        return self.conv(x) # for old versions: double_conv

class TripleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=7, activation="Sigmoid"):
        super().__init__()
        padding = int((kernel_size - 1) / 2)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            getattr(nn, activation)(),
            
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            getattr(nn, activation)(),
            
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            getattr(nn, activation)(),
        )

    def forward(self, x):
        return self.conv(x)

class OutDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) => convolution"""

    def __init__(self, in_channels, out_channels,kernel_size=7, activation="PReLU"):
        super().__init__()
        padding = int((kernel_size - 1) / 2)

        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            getattr(nn, activation)(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, kernel_size)
        self.maxpool = nn.MaxPool1d(2)

        #for old versions 
        # self.maxpool_conv = nn.Sequential(
        #     nn.MaxPool1d(2),
        #     DoubleConv(in_channels, out_channels, kernel_size)
        # )

    def forward(self, x):
        return self.maxpool(self.conv(x))

        # #for old versions
        # return self.maxpool_conv(x)

class Down_with_sc(Down):
    """Downscaling with maxpool then double conv"""
    
    def forward(self, x):
        x1 = self.conv(x)
        return self.maxpool(x1), x1
        

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # self.up = F.interpolate()
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        else:
            self.up = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, kernel_size)

    def forward(self, x1):
        x = self.up(x1)
        return self.conv(x)

class Up_with_sc(Up):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, kernel_size, bilinear=True, concat=True):
        super().__init__(in_channels, out_channels, kernel_size, bilinear)
        self.concat = concat

    def forward(self, x, skip=None):
        if self.concat:
            x1 = torch.cat([skip, x], dim=1)
        else:
            x1 = x
        x2 = self.up(x1)
        return self.conv(x2)

################
# complex blocks
################

class encoder_conv(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.input_dim = args["input_dim"]
        self.latent_dim = args["latent_dim"]
        self.depth = args["input_dim"][0]
        self.n_channels = args["input_dim"][1]
        self.chunk_length = args["input_dim"][2]
        chunk_lenght_after_convs = int(((self.chunk_length - 6 + 1) - 6 + 1)/2) #57, should be checked for consistency with self.chunk_length!

        self.conv_layers = nn.Sequential(
            conv_block(self.depth, 32, kernel=(self.n_channels, 6)),
            conv_block(32, 32, kernel=(1, 6), stride=(1, 2)),
        )

        self.fc = nn.Linear(32*chunk_lenght_after_convs, args["latent_dim"]*args["latent_multiplier"]) #embedding_log_var

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x).reshape(x.shape[0], -1)

        output = self.fc(h1)
        return output

class decoder_conv(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_dim = args["input_dim"]
        self.latent_dim = args["latent_dim"]
        self.depth = args["input_dim"][0]
        self.n_channels = args["input_dim"][1]
        self.chunk_length = args["input_dim"][2]
        self.chunk_lenght_after_convs = int(((self.chunk_length - 6 + 1) - 6 + 1)/2) #57, should be checked for consistency with self.chunk_length!

        self.fc = nn.Linear(args["latent_dim"], 32*self.chunk_lenght_after_convs)
        self.conv_layers = nn.Sequential(
            conv_transpose_block(32, 32, kernel=(1, 6), stride=(1, 2), output_padding=(0, 1)),
            conv_transpose_block(32, self.depth, kernel=(self.n_channels, 6)),
        )

    def forward(self, z: torch.Tensor):
        h1 = self.fc(z).reshape(z.shape[0], 32, 1, self.chunk_lenght_after_convs)
        output = self.conv_layers(h1)

        return output

class encoder_conv_bVAE(torch.nn.Module):
    def __init__(
        self, 
        n_channels
    ):
        super().__init__()

        self.n_channels = n_channels
        self.inc = DoubleConv(n_channels, 4, kernel_size=7) #120
        self.down1 = Down(4, 8, kernel_size=7) #60
        self.down2 = Down(8, 16,kernel_size=5) #30
        self.down3 = Down(16, 64,kernel_size=3) #15

    def forward(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x4

class decoder_conv_bVAE(torch.nn.Module):
    def __init__(
        self, 
        n_classes
    ):
        super().__init__()

        self.n_classes = n_classes
        self.up1 = Up(32, 16, kernel_size=3) #15
        self.up2 = Up(16, 8, kernel_size=3) #30
        self.up3 = Up(8, 4, kernel_size=3) #60
        self.outc = OutConv(4, n_classes, kernel_size=1) #120

    def forward(self, x: torch.Tensor):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        logits = self.outc(x)
        return logits
