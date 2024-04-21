def dummy_callback(x):
    return x

def zero_out_channel(x, channel=None):
    if channel is None:
        return x
    else:
        x[..., channel, :] = 0
        return x