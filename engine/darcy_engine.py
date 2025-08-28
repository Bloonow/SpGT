from typing import Callable

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


def train_epoch_darcy(
    model: torch.nn.Module, train_loader: DataLoader, train_loss_func: Callable, optimizer: Optimizer,
    lr_scheduler: LRScheduler = None, grad_clip=0.99, *, device
):
    model.train()
    loss_epoch = torch.tensor([0.], device=device)
    for data in train_loader:
        node, position, grid = data['node'].to(device), data['position'].to(device), data['grid'].to(device)
        coeff, target, target_grad = data['coeff'].to(device), data['target'].to(device), data['target_grad'].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(node, position=position, grid=grid)
        loss, reg, _ = train_loss_func(pred, target, pred_grad=None, target_grad=target_grad, coeff=coeff)
        loss = loss + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        loss_epoch = loss_epoch + loss
    loss_epoch = loss_epoch / len(train_loader)
    return loss_epoch
    # return loss_epoch.item()


def validate_epoch_darcy(
    model: torch.nn.Module, valid_loader: DataLoader, valid_loss_func: Callable, *, device
):
    model.eval()
    metric_epoch = torch.tensor([0.], device=device)
    with torch.no_grad():
        for data in valid_loader:
            node, position, grid = data['node'].to(device), data['position'].to(device), data['grid'].to(device)
            target = data['target'].to(device)
            pred = model(node, position=position, grid=grid)
            _, _, metric = valid_loss_func(pred, target, pred_grad=None, target_grad=None, coeff=None)
            metric_epoch = metric_epoch + metric
    metric_epoch = metric_epoch / len(valid_loader)
    return metric_epoch
    # return metric_epoch.item()
