import torch
import torch.nn as nn
import typing as t
import numpy as np

###############
# simple blocks
###############

class FlattenLinear(nn.Module):
    def __init__(
        self,
        in_features : int,
        out_features : int,
        **args
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, **args)

    def forward(self, x: torch.Tensor):
        return self.linear(torch.flatten(x, start_dim=1))
    
class UnflattenLinear(nn.Module):
    def __init__(
        self,
        in_features : int,
        out_features : int,
        penultimate_dim: int,
        **args
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, **args)
        self.penultimate_dim = penultimate_dim
        self.out_features = out_features

    def forward(self, x: torch.Tensor):
        return torch.unflatten(self.linear(x), dim=-1, sizes=(self.penultimate_dim, self.out_features//self.penultimate_dim))

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

class NConv(nn.Module):
    """(convolution => [BN] => ReLU) * N"""

    def __init__(
        self, 
        n_convs:int=None, 
        in_channels:int=None, 
        out_channels:t.Union[int, t.List[int]]=None, 
        kernel_size:t.Union[int, t.List[int]]=7, 
        activation:str="Sigmoid", 
        normalize_last:bool=True
    ):
        super().__init__()
        
        assert isinstance(out_channels, int) or len(out_channels)==n_convs, "Length ouf out_channels list should be equal to n_convs"
        assert isinstance(kernel_size, int) or len(kernel_size)==n_convs, "Length ouf kernel_size list should be equal to n_convs"
        
        modules = []
        if isinstance(out_channels, int): out_channels = np.array([out_channels]*n_convs)
        if isinstance(kernel_size, int): kernel_size = np.array([kernel_size]*n_convs)
        padding = (kernel_size - 1)//2
        
        for stage in range(n_convs - 1):
            modules.extend([
                nn.Conv1d(in_channels if stage==0 else out_channels[stage - 1], out_channels[stage], kernel_size=kernel_size[stage], padding=padding[stage]),
                nn.BatchNorm1d(out_channels[stage]),
                getattr(nn, activation)(),
            ])
            
        modules.append(nn.Conv1d(in_channels if n_convs==1 else out_channels[n_convs-2], out_channels[n_convs-1], kernel_size=kernel_size[n_convs-1], padding=padding[n_convs-1]))
        if normalize_last:
            modules.extend([
                nn.BatchNorm1d(out_channels[n_convs-1]),
                getattr(nn, activation)(),
            ])
            
        self.conv = nn.Sequential(*modules) #conv name for backward compatibility (and here should be double_conv for very old versions)

    def forward(self, x):
        return self.conv(x)

class DoubleConv(NConv):
    """
    (convolution => [BN] => ReLU) * 2
    Left for backward compatibility, all new code should use NConv
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, activation="Sigmoid"):
        super().__init__(2, in_channels, out_channels, kernel_size=kernel_size, activation=activation, normalize_last=True)

class TripleConv(NConv):
    """
    (convolution => [BN] => ReLU) * 3
    Left for backward compatibility, all new code should use NConv
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, activation="Sigmoid"):
        super().__init__(3, in_channels, out_channels, kernel_size=kernel_size, activation=activation, normalize_last=True)

# class OutConv(NConv):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super().__init__(1, in_channels, out_channels, kernel_size=kernel_size, normalize_last=False)
    
class OutConv(nn.Module):
    """
    Left for backward compatibility, all new code should use NConv.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(OutConv, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x):
        return self.conv(x)
        

class OutDoubleConv(NConv):
    """
    (convolution => [BN] => ReLU) => convolution
    Left for backward compatibility, all new code should use NConv.
    Previous version used self.double_conv instead of conv.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init(2, in_channels, out_channels, kernel_size=kernel_size, normalize_last=False)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        if dim % num_heads:
            raise ValueError('dim % num_heads != 0')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        '''
        Args: 
            x: Tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Tensor of shape (batch_size, seq_len, input_dim)
        '''
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # qkv: 3 × B × num_heads × N × head_dim
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v  # attn: B × num_heads × N × N    v: B × num_heads × N × head_dim -> B × num_heads × N × head_dim
        x = x.transpose(1, 2).reshape(B, N, C) # B × N × (num_heads × head_dim)
        x = self.proj(x)
        return x

################
# complex blocks
################

class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, n_convs=2, activation="Sigmoid", normalize_last=True):
        super().__init__()
        self.conv = NConv(n_convs, in_channels, out_channels, kernel_size=kernel_size, activation=activation, normalize_last=normalize_last)
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

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, bilinear=True, n_convs=2, activation="Sigmoid", normalize_last=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # self.up = F.interpolate()
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        else:
            self.up = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = NConv(n_convs, in_channels, out_channels, kernel_size=kernel_size, activation=activation, normalize_last=normalize_last)

    def forward(self, x1):
        x = self.up(x1)
        return self.conv(x)


class Up_with_sc(Up):
    """Upscaling then double conv"""
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, bilinear=True, concat=True, n_convs=2, activation="Sigmoid", normalize_last=True):
        super().__init__(in_channels, out_channels, kernel_size, bilinear, n_convs, activation=activation, normalize_last=normalize_last)
        self.concat = concat

    def forward(self, x, skip=None):
        if self.concat:
            x1 = torch.cat([skip, x], dim=1)
        else:
            x1 = x
        x2 = self.up(x1)
        return self.conv(x2)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim:int=None,
        num_heads:int=None,
        mlp_ratio:int=4,  # ratio between hidden_dim and input_dim in MLP,
        mlp_depth:int=2,
        p_dropout:float=0.1,
        act_layer:str="PReLU",
        norm_layer:str="LayerNorm"
    ):
        super().__init__()
        
        self.norm1 = getattr(nn, norm_layer)(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads)
        self.norm2 = getattr(nn, norm_layer)(dim)
        
        mlp_layers = []
        for i in range(mlp_depth):
            mlp_layers.extend([
                nn.Linear(dim * mlp_ratio if i != 0 else dim, dim * mlp_ratio if i != mlp_depth-1 else dim), 
                nn.Dropout(p=p_dropout),
            ])
            if i != mlp_depth-1: mlp_layers.append(getattr(nn, act_layer)())
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.mlp(x))
        return x


class DownTransformer(nn.Module):
    """
    Downscaling with maxpool then transformer
    """

    def __init__(
        self,
        input_dim:int=128,
        seq_length:int=3,
        num_heads:int=1,
        depth:int=1,
        mlp_depth:int=2,
        mlp_ratio:int=4,
        p_dropout:float=0.1,
        act_layer:str='PReLU',
        norm_layer:str='LayerNorm',
        with_outro:bool=False,
    ):
        super().__init__()

        self.dim = input_dim
        self.seq_length = seq_length
        self.with_outro = with_outro
        self.num_heads = num_heads
        self.head_dim = self.dim//num_heads
        
        self.maxpool = nn.MaxPool1d(2)
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_length, self.dim) * .02)
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=self.dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                mlp_depth=mlp_depth,
                p_dropout=p_dropout, 
                act_layer=act_layer, 
                norm_layer=norm_layer,
            ) for _ in range(depth)
        ])
        if self.with_outro: 
            self.outro = [nn.Linear(self.dim*self.seq_length//num_heads, self.dim*self.seq_length//num_heads) for _ in range(self.num_heads)]
    
    def forward(self, x): #B, C, N
        x += self.pos_embed #B, C, N
        x = self.blocks(x) #B, C, N
        if self.with_outro:
            x = x.reshape(-1, self.seq_length, self.num_heads, self.head_dim)
            x = x.permute(2, 0, 1, 3).reshape(self.num_heads, -1, self.seq_length*self.head_dim)
            x = torch.stack([proj(inp) for proj, inp in zip(self.outro, x)], dim=0)
            x = x.reshape(self.num_heads, -1, self.seq_length, self.head_dim).permute(1, 2, 0, 3)
            x = x.reshape(-1, self.seq_length, self.dim)

        x = self.maxpool(x) #B, C, N//2
        return x


class UpTransformer(nn.Module):
    """Upscaling then transformer"""

    def __init__(
        self,
        input_dim:int=128,
        seq_length:int=3,
        num_heads:int=1,
        depth:int=1,
        mlp_depth:int=2,
        mlp_ratio:int=4,
        p_dropout:float=0.1,
        act_layer:str='PReLU',
        norm_layer:str='LayerNorm',
        with_outro:bool=False,
        bilinear=True,
    ):
        super().__init__()

        self.dim = input_dim*2
        self.seq_length = seq_length
        self.with_outro = with_outro
        self.num_heads = num_heads
        self.num_heads = num_heads
        self.head_dim = self.dim//num_heads

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear: self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        else: self.up = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_length, self.dim) * .02)
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=self.dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                mlp_depth=mlp_depth,
                p_dropout=p_dropout, 
                act_layer=act_layer, 
                norm_layer=norm_layer,
            ) for _ in range(depth)
        ])
        if self.with_outro: 
            self.outro = [nn.Linear(self.dim*self.seq_length//num_heads, self.dim*self.seq_length//num_heads) for _ in range(self.num_heads)]
    
    def forward(self, x): #B, C, N
        x = self.up(x) #B, C, N*2
        x += self.pos_embed #B, C, N*2
        x = self.blocks(x) #B, C, N*2
        if self.with_outro:
            x = x.reshape(-1, self.seq_length, self.num_heads, self.head_dim)
            x = x.permute(2, 0, 1, 3).reshape(self.num_heads, -1, self.seq_length*self.head_dim)
            x = torch.stack([proj(inp) for proj, inp in zip(self.outro, x)], dim=0)
            x = x.reshape(self.num_heads, -1, self.seq_length, self.head_dim).permute(1, 2, 0, 3)
            x = x.reshape(-1, self.seq_length, self.dim)
        return x

#########
# subnets
#########

class encoder_conv(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.input_dim = args["input_dim"]
        self.latent_dim = args["latent_dim"]
        self.depth = args["input_dim"][0]
        self.n_channels = args["input_dim"][1]
        self.chunk_length = args["input_dim"][2]
        chunk_lenght_after_convs = int(((self.chunk_length - 6 + 1) - 6 + 1)/2) #if chunk_length=14, chunk_lenght_after_convs=57, should be checked for consistency with self.chunk_length!

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
        

class ConvEncoder(nn.Module):
    def __init__(
        self,
        down_blocks_config:t.List[t.Dict] = None,
        out_conv_config:t.Optional[t.Dict] = None
    ):
        super().__init__()
        modules = []
        for params in down_blocks_config: modules.append(Down(**params))
        if out_conv_config is not None: modules.append(NConv(**out_conv_config))
        self.conv = nn.Sequential(*modules)

    def forward(self, x:torch.Tensor):
        return self.conv(x)

class ConvDecoder(nn.Module):
    def __init__(
        self,
        up_blocks_config:t.List[t.Dict] = None,
        in_conv_config:t.Optional[t.Dict] = None
    ):
        super().__init__()
        modules = []
        if in_conv_config is not None: modules.append(NConv(**in_conv_config))
        for params in up_blocks_config: modules.append(Up(**params))
        self.conv = nn.Sequential(*modules)

    def forward(self, x:torch.Tensor):
        return self.conv(x)
        

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        down_blocks_config:t.List[t.Dict] = None,
        out_block_config:t.Optional[t.Dict] = None
    ):
        super().__init__()
        modules = []
        for params in down_blocks_config: modules.append(DownTransformer(**params))
        if out_block_config is not None: modules.append(TransformerBlock(**out_block_config))
        self.blocks = nn.Sequential(*modules)

    def forward(self, x:torch.Tensor):
        return self.blocks(x)

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        up_blocks_config:t.List[t.Dict] = None,
        in_block_config:t.Optional[t.Dict] = None
    ):
        super().__init__()
        modules = []
        if in_block_config is not None: modules.append(TransformerBlock(**in_block_config))
        for params in up_blocks_config: modules.append(UpTransformer(**params))
        self.blocks = nn.Sequential(*modules)

    def forward(self, x:torch.Tensor):
        return self.blocks(x)