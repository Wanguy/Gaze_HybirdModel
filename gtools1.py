import torch
import numpy as np

# torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=18)


def gazeto3d(gaze):
    assert gaze.shape == torch.Size([2]), "The size of gaze must be 2"
    gaze_gt = torch.zeros(3, 1)
    gaze_gt[0] = -torch.cos(gaze[1]) * torch.sin(gaze[0])
    gaze_gt[1] = -torch.sin(gaze[1])
    gaze_gt[2] = -torch.cos(gaze[1]) * torch.cos(gaze[0])
    return gaze_gt


def angular(gaze, label):
    assert gaze.shape == torch.Size([3, 1]), "The size of gaze must be 3"
    assert label.shape == torch.Size([3, 1]), "The size of label must be 3"

    total = torch.sum(torch.mul(gaze, label))
    return (torch.arccos(torch.min(total / (torch.norm(gaze) * torch.norm(label)), torch.tensor(.9999999))) * 180 / np.pi).item()
