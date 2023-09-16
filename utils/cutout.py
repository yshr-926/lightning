import torch

def Cutout(img, n_holes, length):
    half_length = length // 2

    left = torch.randint(-half_length, img.size(-1) - half_length, [1]).item()
    top = torch.randint(-half_length, img.size(-2) - half_length, [1]).item()
    right = min(img.size(-1), left + length)
    bottom = min(img.size(-2), top + length)

    img[..., max(0, left): right, max(0, top): bottom] = 0
    return img
