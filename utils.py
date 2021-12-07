import wandb

import random

import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torchvision.transforms.functional as F


class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)
    
    
def data_preprocessing(input, sigma):
        sigma = sigma / 255.0
        noise = torch.zeros_like(input)
        noise.normal_(0, 1)
        noise *= sigma
        noisy = input + noise

        return noisy, input

def criterion(pred, targets, patch_size=10):
    lossfac = 1600.0 / patch_size**2
    loss = (0.5*(pred-targets)**2).reshape(pred.shape[0],-1).sum(dim=1, keepdim=True) * lossfac
    return loss.mean()

@torch.no_grad()
def grad_check(named_parameters):
    thresh = 0.001

    layers = []
    max_grads = []
    mean_grads = []
    max_colors = []
    mean_colors = []

    for n, p in named_parameters:
        # import pdb; pdb.set_trace()
        # print(n)
        if not p.requires_grad:
            print(n)
        elif p.requires_grad and "bias" not in n:
            max_grad = p.grad.abs().max()
            mean_grad = p.grad.abs().mean()
            layers.append(n)
            max_grads.append(max_grad)
            mean_grads.append(mean_grad)

    for i, (val_mx, val_mn) in enumerate(zip(max_grads, mean_grads)):
        if val_mx > thresh:
            max_colors.append("r")
        else:
            max_colors.append("g")
        if val_mn > thresh:
            mean_colors.append("b")
        else:
            mean_colors.append("y")
    ax = plt.subplot(111)
    x = np.arange(len(layers))
    w = 0.3

    ax.bar(x - w, max_grads, width=w, color=max_colors, align="center", hatch="////")
    ax.bar(x, mean_grads, width=w, color=mean_colors, align="center", hatch="----")

    plt.xticks(x - w / 2, layers, fontsize=4, rotation="vertical")
    plt.xlim(left=-1, right=len(layers))
    # plt.ylim(bottom=0.0, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Gradient Values")
    plt.title("Model Gradients")

    hatch_dict = {0: "////", 1: "----"}
    legends = []
    for i in range(len(hatch_dict)):
        p = patches.Patch(facecolor="#DCDCDC", hatch=hatch_dict[i])
        legends.append(p)

    ax.legend(legends, ["Max", "Mean"])

    plt.grid(True)
    plt.tight_layout()
    wandb.log({"Gradients": wandb.Image(plt)})
    plt.close()
