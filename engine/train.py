from typing import Callable
from termcolor import colored

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from SpGT.common.trivial import is_main_process


def run_train(
    model: torch.nn.Module, train_loader: DataLoader, valid_loader: DataLoader,
    train_epoch_func: Callable, validate_epoch_func: Callable, train_loss_func: Callable, valid_loss_func: Callable,
    optimizer: Optimizer, lr_scheduler: LRScheduler, grad_clip=0.999,
    epochs=10, start_epoch=0, patience=None, use_tqdm_bar=True, *, device
):
    # 判断学习率调度器是按 epoch 修改还是按 batch 修改
    epoch_scheduler_list = ['ReduceLROnPlateau', 'StepLR', 'MultiplicativeLR', 'MultiStepLR', 'ExponentialLR', 'LambdaLR']
    is_epoch_scheduler = any(name in str(lr_scheduler.__class__) for name in epoch_scheduler_list)
    patience = epochs if patience is None else patience
    stop_counter = 0
    best_epoch = start_epoch
    best_metric = float('inf')

    # 训练过程的记录
    checkpoint = dict(loss_history=list(), metric_history=list())
    # 进度条信息
    tqdm_bar = tqdm(total=epochs, disable=not use_tqdm_bar)
    for epoch in range(start_epoch, start_epoch + epochs):
        # 训练一个 epoch
        loss = train_epoch_func(
            model, train_loader, train_loss_func, optimizer,
            None if is_epoch_scheduler else lr_scheduler, grad_clip, device=device
        ).item()
        # 验证一个 epoch
        metric = validate_epoch_func(model, valid_loader, valid_loss_func, device=device).item()
        # 将 loss 与 metric 添加到 checkpoint
        checkpoint['loss_history'].append(loss)
        checkpoint['metric_history'].append(metric)

        # 尝试更新当前 epoch 学习率
        if is_epoch_scheduler:
            if 'ReduceLROnPlateau' in str(lr_scheduler.__class__):
                lr_scheduler.step(metric)
            else:
                lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        # 判断该轮训练是否使目标函数下降
        if metric < best_metric:
            best_epoch, best_metric = epoch, metric
            stop_counter = 0
            # 记录当前效果最好的 epoch 以及模型
            checkpoint['Best_Epoch'] = best_epoch
            checkpoint['Best_Metric'] = best_metric
            checkpoint['Module'] = model.state_dict()
        else:
            stop_counter = stop_counter + 1

        # 是否提前终止
        if stop_counter > patience:
            print(f'Early stop at epoch {epoch}')
            break

        # 进度条更新
        description = colored(f'loss: {loss:.3e}', color='green')
        description += colored(f' | metric: {metric:.3e}', color='blue')
        description += colored(f' | best metric: {best_metric:.3e} at epoch {best_epoch+1}', color='yellow')
        description += colored(f' | early stop: {stop_counter}', color='red')
        description += colored(f' | current lr: {lr:.3e}', color='magenta')
        tqdm_bar.set_description(description)
        tqdm_bar.update()
        # print(description)
    return checkpoint


def run_train_ddp(
    model: torch.nn.Module, train_loader: DataLoader, valid_loader: DataLoader,
    train_sampler: DistributedSampler, valid_sampler: DistributedSampler,
    train_epoch_func: Callable, validate_epoch_func: Callable, train_loss_func: Callable, valid_loss_func: Callable,
    optimizer: Optimizer, lr_scheduler: LRScheduler, grad_clip=0.999,
    epochs=10, start_epoch=0, patience=None, use_tqdm_bar=True, *, device
):
    # 判断学习率调度器是按 epoch 修改还是按 batch 修改
    epoch_scheduler_list = ['ReduceLROnPlateau', 'StepLR', 'MultiplicativeLR', 'MultiStepLR', 'ExponentialLR', 'LambdaLR']
    is_epoch_scheduler = any(name in str(lr_scheduler.__class__) for name in epoch_scheduler_list)
    patience = epochs if patience is None else patience
    stop_counter = 0
    best_epoch = start_epoch
    best_metric = float('inf')

    # 训练过程的记录
    checkpoint = dict(loss_history=list(), metric_history=list())
    # 只有主进程打印进度条信息
    use_tqdm_bar = use_tqdm_bar and is_main_process()
    tqdm_bar = tqdm(total=epochs, disable=not use_tqdm_bar)
    for epoch in range(start_epoch, start_epoch + epochs):
        # 更新数据集采样器
        if train_sampler is not None: train_sampler.set_epoch(epoch)
        if valid_sampler is not None: valid_sampler.set_epoch(epoch)
        # 训练一个 epoch
        loss = train_epoch_func(
            model, train_loader, train_loss_func, optimizer,
            None if is_epoch_scheduler else lr_scheduler, grad_clip, device=device
        ).item()
        # 验证一个 epoch
        metric = validate_epoch_func(model, valid_loader, valid_loss_func, device=device).item()
        # 将 loss 与 metric 添加到 checkpoint
        checkpoint['loss_history'].append(loss)
        checkpoint['metric_history'].append(metric)

        # 尝试更新当前 epoch 学习率
        if is_epoch_scheduler:
            if 'ReduceLROnPlateau' in str(lr_scheduler.__class__):
                lr_scheduler.step(metric)
            else:
                lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        # 判断该轮训练是否使目标函数下降
        if metric < best_metric:
            best_epoch, best_metric = epoch, metric
            stop_counter = 0
            # 记录当前效果最好的 epoch 以及模型
            checkpoint['Best_Epoch'] = best_epoch
            checkpoint['Best_Metric'] = best_metric
            checkpoint['Module'] = model.state_dict()
        else:
            stop_counter = stop_counter + 1

        # 是否提前终止
        if stop_counter > patience:
            print(f'Early stop at epoch {epoch}')
            break

        # 进度条更新
        description = colored(f'loss: {loss:.3e}', color='green')
        description += colored(f' | metric: {metric:.3e}', color='blue')
        description += colored(f' | best metric: {best_metric:.3e} at epoch {best_epoch+1}', color='yellow')
        description += colored(f' | early stop: {stop_counter}', color='red')
        description += colored(f' | current lr: {lr:.3e}', color='magenta')
        tqdm_bar.set_description(description)
        tqdm_bar.update()
        # print(description)
    return checkpoint
