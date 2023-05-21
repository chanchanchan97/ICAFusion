import torch
import numpy as np
#import matlab.engine
import torch.nn.functional as F
from PIL import Image
import scipy.io as sio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denseCFOG(image):
    s_im = image.shape
    if len(s_im) == 2:
        des_tensor = denseCFOG2D(image)
    elif len(s_im) == 3:
        assert s_im[1] == 1
        image = image.squeeze(0)
        des_tensor = denseCFOG2D(image)
    elif len(s_im) == 4:
        batchSize = s_im[0]
        assert s_im[1] == 1
        des_tensor = torch.zeros(batchSize, 9, s_im[2], s_im[3]).to(device)
        for b in range(batchSize):
            des_tensor[b] = denseCFOG2D(image[b].squeeze(0))
    else:
        des_tensor = 0
    return des_tensor

def denseCFOG2D(image):
    eng = matlab.engine.start_matlab()
    eng.cd('./descriptor', nargout=0)
    im_np = np.array(image.detach().cpu())
    im_np = (im_np * 255).astype(np.uint8)
    im_matlab = matlab.uint8(im_np.tolist())
    des_matlab = eng.CFOG_matlab(im_matlab)
    des_np = np.array(des_matlab)
    des_tensor = torch.tensor(des_np, dtype=torch.float32).to(device).permute(2, 0, 1)
    eng.exit()
    return des_tensor
