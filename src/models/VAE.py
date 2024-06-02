import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

def reparameterize(mean, log_std):
    return mean + torch.randn_like(mean) * torch.exp(log_std)

def kl(mean, log_std, reduce_mode="sum"):
    kl_loss = ((torch.exp(2 * log_std) + mean * mean) / 2 - 0.5 - log_std)
    if reduce_mode == "sum": return kl_loss.sum(axis=1)
    else: return kl_loss.mean(axis=1)

class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, **args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.Z_DIM = args["latent_dim"]
        self.beta = args.get("beta", 1)
        self.first_decoder_conv_depth = args.get("first_decoder_conv_depth", None)
        self.loss_reduction = args.get("loss_reduction", "mean") #sum or mean

        if self.loss_reduction not in ["mean", "sum"]:
            raise NotImplementedError(f"Unsupported loss reduce mode {self.loss_reduction}")
        
    def _encode(self, imgs):
        z_params = self.encoder(imgs).reshape(imgs.shape[0], -1)
        z_mean, z_log_std = torch.split(z_params, [self.Z_DIM, self.Z_DIM], dim=1)
        return z_mean, z_log_std

    def encode(self, imgs):
        # return self._encode(imgs)[0]
        return self.encoder(imgs)

    def decode(self, z):
        return self.decoder(z)

    def _reconstruct(self, imgs):
        z_mean, z_log_std = self._encode(imgs)
        z = reparameterize(z_mean, z_log_std)
        if self.first_decoder_conv_depth is not None: z = z.reshape(imgs.shape[0], self.first_decoder_conv_depth, -1) #for models with Conv1d
        decoded_imgs = self.decode(z)
        return decoded_imgs, z_mean, z_log_std
    
    def reconstruct(self, imgs):
        return self._reconstruct(imgs)[0]

    def forward(self, imgs):
        decoded_imgs, z_mean, z_log_std = self._reconstruct(imgs)

        z_kl = kl(z_mean, z_log_std, reduce_mode=self.loss_reduction)
        reduce_dims = list(range(len(imgs.shape)))[1:]
        if self.loss_reduction == "sum": err = ((imgs - decoded_imgs)**2).sum(reduce_dims) #for 4-layer beta-VAE 
        elif self.loss_reduction == "mean" : err = ((imgs - decoded_imgs)**2).mean(reduce_dims)
        else: raise NotImplementedError(f"Unsupported loss reduce mode {self.loss_reduction}")
        # log_p_x_given_z = -err
        # loss = -log_p_x_given_z + self.beta*z_kl
        loss = err + self.beta*z_kl

        return {
            'loss': loss.mean(),
            '-log p(x|z)': err.mean(),
            'kl': z_kl.mean(),
            'decoded_imgs': decoded_imgs
        }

#####################################
# Burgess and Higgins implementations
# code from https://github.com/1Konny/Beta-VAE/blob/master
#####################################

def reconstruction_loss(x, x_recon, distribution, reduce_mode="mean"):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False, reduction=reduce_mode).div(batch_size)
    elif distribution == 'gaussian':
#         x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False, reduction=reduce_mode).div(batch_size)
    else:
        raise ValueError("Unknown distribution")

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4: mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4: logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def reparametrize_BH(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

# def kaiming_init(m):
#     if isinstance(m, (nn.Linear, nn.Conv2d)):
#         init.kaiming_normal(m.weight)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#     elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#         m.weight.data.fill_(1)
#         if m.bias is not None:
#             m.bias.data.fill_(0)


# def normal_init(m, mean, std):
#     if isinstance(m, (nn.Linear, nn.Conv2d)):
#         m.weight.data.normal_(mean, std)
#         if m.bias.data is not None:
#             m.bias.data.zero_()
#     elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
#         m.weight.data.fill_(1)
#         if m.bias.data is not None:
#             m.bias.data.zero_()


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, encoder, decoder, **args):
        super(BetaVAE_H, self).__init__()
        
        self.z_dim = args["latent_dim"]
        self.beta = args["beta"]
        self.decoder_dist = "gaussian" #"gaussian"/"bernoulli"
        self.loss_reduction = args["loss_reduction"]
        
        self.encoder = encoder
        self.decoder = decoder

#         self.weight_init()

#     def weight_init(self):
#         for block in self._modules:
#             for m in self._modules[block]:
#                 kaiming_init(m)

    def forward(self, x):
        x_recon, mu, logvar = self._reconstruct(x)
        
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist, reduce_mode=self.loss_reduction)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        if self.loss_reduction == "sum": kl = total_kld
        elif self.loss_reduction == "mean": kl == mean_kld
        
        beta_vae_loss = recon_loss + self.beta*kl

        return {
            'loss': beta_vae_loss,
            '-log p(x|z)': recon_loss,
            'kl': total_kld,
            'decoded_imgs': x_recon
        }

    def _encode(self, x):
        return self.encoder(x)
    
    def encode(self, x):
        distributions = self._encode(x)
        return distributions[:, :self.z_dim]

    def _decode(self, z):
        return self.decoder(z)
    
    def _reconstruct(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize_BH(mu, logvar)
        x_recon = self._decode(z)
        return x_recon, mu, logvar
    
    def reconstruct(self, x):
        x_recon, mu, logvar = self._reconstruct(x)
        return x_recon


class BetaVAE_B(BetaVAE_H):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, encoder, decoder, **args):
        super(BetaVAE_B, self).__init__(encoder, decoder, **args)
        self.global_iter = 0 
        self.C_max = Variable(torch.FloatTensor([args["C_max"]])).to(args["device"]) #20
        self.C_stop_iter = args["C_stop_iter"] #1e5
        
#         self.weight_init()

    def forward(self, x):
        x_recon, mu, logvar = self._reconstruct(x)

        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist, reduce_mode=self.loss_reduction)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        if self.loss_reduction == "sum": kl = total_kld
        elif self.loss_reduction == "mean": kl == mean_kld
        
        C = self.get_C()
        beta_vae_loss = recon_loss + self.beta*(kl - C).abs()

        return {
            'loss': beta_vae_loss,
            '-log p(x|z)': recon_loss,
            'kl': total_kld,
            'decoded_imgs': x_recon
        }
    
    def get_C(self):
        return torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
            
    def reset_global_iter(self):
        self.global_iter = 0
        
    def update_global_iter(self, n=1):
        self.global_iter += n