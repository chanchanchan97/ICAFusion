import torch
import numpy as np
#import matlab.engine
#eng = matlab.engine.start_matlab()
#eng.cd('./descriptor',nargout=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denseLSS(image):
    s_im = image.shape
    if len(s_im) == 2:
        des_tensor = denseLSS2D(image)
    elif len(s_im) == 3:
        assert s_im[1] == 1
        image = image.squeeze(0)
        des_tensor = denseLSS2D(image)
    elif len(s_im) == 4:
        batchSize = s_im[0]
        assert s_im[1] == 1
        des_tensor = torch.zeros(batchSize, 18, s_im[2], s_im[3]).to(device)
        for b in range(batchSize):
            des_tensor[b] = denseLSS2D(image[b].squeeze(0))
    else:
        des_tensor = 0
    return des_tensor

def denseLSS2D(image):
    im_np = np.array(image.detach().cpu())
    im_np = (im_np * 255).astype(np.uint8)
    im_matlab = matlab.uint8(im_np.tolist())
    des_matlab = eng.denseLSS(im_matlab, 3.0, 2.0, 9.0)
    des_np = np.array(des_matlab)
    des_tensor = torch.tensor(des_np, dtype=torch.float32).to(device)
    return des_tensor

def denseLSS_matlab(image):
    s_im = image.shape
    if len(s_im) == 2:
        des_matlab = denseLSS2D_matlab(image)
    elif len(s_im) == 3:
        assert s_im[1] == 1
        image = image.squeeze(0)
        des_matlab = denseLSS2D_matlab(image)
    elif len(s_im) == 4:
        batchSize = s_im[0]
        assert s_im[1] == 1
        des_matlab = eng.zeros(batchSize, 18, s_im[2], s_im[3])
        for b in range(batchSize):
            des_matlab[b] = denseLSS2D_matlab(image[b].squeeze(0))
    else:
        des_matlab = 0
    return des_matlab

def denseLSS2D_matlab(image):
    im_np = np.array(image.detach().cpu())
    im_np = (im_np * 255).astype(np.uint8)
    im_matlab = matlab.uint8(im_np.tolist())
    des_matlab = eng.denseLSS(im_matlab, 3.0, 2.0, 9.0)
    return des_matlab