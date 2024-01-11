import torch
import torch.nn as nn

from .modules import DoubleConv, Down, Up, OutConv

class AE(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(AE, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 4, kernel_size=7) #1024
        self.down1 = Down(4, 8, kernel_size=7) #512
        self.down2 = Down(8, 16,kernel_size=5) #256
        self.down3 = Down(16, 32,kernel_size=3) #128
        self.up1 = Up(32, 16, kernel_size=3) #256
        self.up2 = Up(16, 8, kernel_size=3) #512
        self.up3 = Up(8, 4, kernel_size=3) #1024
        self.outc = OutConv(4, n_classes, kernel_size=1) #1024

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4)
        x = self.up2(x)
        x = self.up3(x)
        logits = self.outc(x)
        return logits
    
    def reconstruct(self, x):
        return self.forward(x)
    
    def encode(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x4

class AE_framework(torch.nn.Module):
    def __init__(self, encoder, decoder, **args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.Z_DIM = args["latent_dim"]
        self.input_dim = args["input_dim"]
        self.first_decoder_conv_depth = args["first_decoder_conv_depth"] #also affect loss redicing
        self.loss_reduce_mode = args["loss_reduce_mode"] #sum or mean
        
    def _encode(self, imgs):
        return self.encoder(imgs)

    def encode(self, imgs):
        return self._encode(imgs)

    def decode(self, z):
        return self.decoder(z)

    def _reconstruct(self, imgs):
        z = self._encode(imgs)
        if self.first_decoder_conv_depth is not None: z = z.reshape(imgs.shape[0], self.first_decoder_conv_depth, -1) #for 4-layer beta-VAE
        decoded_imgs = self.decode(z)
        return decoded_imgs, z
    
    def reconstruct(self, imgs):
        return self._reconstruct(imgs)[0]

    def forward(self, imgs):
        decoded_imgs, z = self._reconstruct(imgs)

        if self.first_decoder_conv_depth is not None: reduce_dims = [1, 2]
        else: reduce_dims = [1, 2, 3]
        if self.loss_reduce_mode == "sum": err = ((imgs - decoded_imgs)**2).sum(reduce_dims) #for 4-layer beta-VAE 
        elif self.loss_reduce_mode == "mean" : err = ((imgs - decoded_imgs)**2).mean(reduce_dims)
        else: raise NotImplementedError(f"Unsupported loss reduce mode {self.loss_reduce_mode}")
        log_p_x_given_z = -err
        loss = -log_p_x_given_z

        return {
            'loss': loss.mean(),
            '-log p(x|z)': -log_p_x_given_z.mean(),
            'decoded_imgs': decoded_imgs
        }