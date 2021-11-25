import numpy as np 

def format_evenodd(data):
    # Formats data into arrya of even and odd channels
    # (B, H, W, C) -> (B, H, W, C/2) 
    
    x = data[:,:,:,0::2] # Even
    y = data[:,:,:,1::2] # Odd
    x = format_to_complex(x)
    y = format_to_complex(y)
    return x, y

def format_to_complex(data):
    # Formats data by spliting up into real/imag parts
    # (B, H, W, C) -> (B, H, W, 2 * C) 

    s = data.shape
    _data = np.zeros((s[0], s[1], s[2], 2*s[3]))
    for n in range(s[3]):
        _data[:,:,:,2*n] = data[:,:,:,n].real
        _data[:,:,:,2*n+1] = data[:,:,:,n].imag
    return _data