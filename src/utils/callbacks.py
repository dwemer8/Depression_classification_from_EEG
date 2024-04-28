import torch

def dummy_callback(x):
    return x

def zero_out_channel(x, channel=None):
    if channel is None:
        return x
    else:
        x[..., channel, :] = 0
        return x
    
def mask_chunks(chunks, mask_ratio = 0.5):
    if mask_ratio == 0:
        return chunks
        
    B = chunks.shape[0]
    length = chunks.shape[-1]
    mask_length = int(length*mask_ratio)
    max_mask_idx = length - mask_length
    mask_idxs = (max_mask_idx * torch.rand(*list(chunks.shape[:-1]))).type(torch.int)
    
    chunks = chunks.clone().detach()
    chunks_mean = chunks.mean(dim=-1, keepdim=True)
    chunks_std = chunks.std(dim=-1, keepdim=True)
    noise = torch.randn(*chunks.shape[:-1], mask_length)*chunks_std + chunks_mean
    
    if (noise.max() > 1 or noise.min() < 0) and noise.max() - noise.min() != 0:
            noise = (noise - noise.min())/(noise.max() - noise.min()) 
            
    if len(chunks.shape) == 3:
        for chunk, noise_chunk, chunk_mask_idxs in zip(chunks, noise, mask_idxs):
            for ch, noise_ch, ch_mask_idx in zip(chunk, noise_chunk, chunk_mask_idxs):
                ch[ch_mask_idx:ch_mask_idx + mask_length] = noise_ch
                
    elif len(chunks.shape) == 4:
        for chunk, noise_chunk, chunk_mask_idxs in zip(chunks, noise, mask_idxs):
            for ch_d, noise_ch_d, ch_d_mask_idxs in zip(chunk, noise_chunk, chunk_mask_idxs):
                for ch, noise_ch, ch_mask_idx in zip(ch_d, noise_ch_d, ch_d_mask_idxs):
                    ch[ch_mask_idx:ch_mask_idx + mask_length] = noise_ch
                    
    else:
        raise ValueError(f"Unexpected number of dimensions in chunks: {len(chunks.shape)}")
                
    return chunks