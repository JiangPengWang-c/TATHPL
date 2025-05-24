# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import numpy as np
import pandas as pd
import torch.nn as nn
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import  TPLoss
import utils

import gc


def one_hot(x_1, x_2, num_classes, on_value_1, on_value_2, off_value=0.1, device='cuda'):
    x_1 = x_1.long().view(-1, 1)
    x_2 = x_2.long().view(-1, 1)
    # print(x_1 == x_2)
    return torch.full(
        (x_1.size()[0], num_classes), \
        off_value / num_classes, \
        device=device).scatter_ \
        (1, x_1, on_value_1 - off_value / 2 + off_value / num_classes).scatter_ \
        (1, x_2, on_value_2 - off_value / 2 + off_value / num_classes) * (x_1 != x_2) + \
        torch.full(
            (x_1.size()[0], num_classes), \
            off_value / num_classes, \
            device=device).scatter_ \
            (1, x_1, on_value_1 + on_value_2 - off_value + off_value / num_classes) * (x_1 == x_2)


# voc2007
# def train_one_epoch(model: torch.nn.Module, criterion: TPLoss,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler,scheduler, max_norm: float = 0,
#                     model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
#                     set_training_mode=True, args=None):

#     model.train(set_training_mode)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 20

#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
#         samples = samples.to(device, non_blocking=True)

#         with torch.cuda.amp.autocast():
#             outputs_list = model(samples, targets)
#             #loss ,,
#             loss,sig_loss0,sig_loss1,base_loss= criterion(outputs_list, targets)
#         base_loss_value = base_loss.item()
#         sig_loss0_value = sig_loss0.item()
#         sig_loss1_value = sig_loss1.item()
#         # sig_loss2_value = sig_loss2.item()
#         # sig_loss3_value = sig_loss3.item()sig_loss2,sig_loss3,
#         loss_value = loss.item()

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             sys.exit(1)
#         optimizer.zero_grad()
#         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#         loss_scaler(loss, optimizer, clip_grad=max_norm,
#                     parameters=model.parameters(), create_graph=is_second_order)
#         scheduler.step()
#         torch.cuda.synchronize()
#         if model_ema is not None:
#             model_ema.update(model)
#         metric_logger.update(loss=loss_value)
#         metric_logger.update(base_loss=base_loss_value)
#         metric_logger.update(sig_loss0=sig_loss0_value)
#         metric_logger.update(sig_loss1=sig_loss1_value)
#         # metric_logger.update(sig_loss2=sig_loss2_value)
#         # metric_logger.update(sig_loss3=sig_loss3_value)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# ###for coco
# def train_one_epoch(model: torch.nn.Module, criterion: TPLoss,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler,scheduler,max_norm: float = 0,
#                     model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
#                     set_training_mode=True, args=None):

#     model.train(set_training_mode)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 100

#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
#         samples = samples.to(device, non_blocking=True)
# #         target=targets[0]
# #         topic=targets[1]

# #         target = target.cuda()
# #         target = target.max(dim=1)[0]
# #         target_list=[]
# #         target_list.append(target)
# #         target_list.append(topic)

#         with torch.cuda.amp.autocast():
#             outputs_list = model(samples,targets)
#             base_loss,sig_loss0,sig_loss1,loss= criterion(outputs_list, targets)
#         loss_value = loss.item()
#         base_loss_value = base_loss.item()
#         sig_loss0_value = sig_loss0.item()
#         sig_loss1_value = sig_loss1.item()
#         #sig_loss2_value = sig_loss2.item()sig_loss2,
#         #sig_loss3_value = sig_loss3.item()sig_loss3,
#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             sys.exit(1)
#         optimizer.zero_grad()
#         # this attribute is added by timm on one optimizer (adahessian)
#         # Wait for update
#         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#         loss_scaler(loss, optimizer, clip_grad=max_norm,
#                    parameters=model.parameters(), create_graph=is_second_order)
#         scheduler.step()
#         torch.cuda.synchronize()
#         if model_ema is not None:
#             model_ema.update(model)

#         metric_logger.update(loss=loss_value)
#         metric_logger.update(base_loss=base_loss_value)
#         metric_logger.update(sig_loss0=sig_loss0_value)
#         metric_logger.update(sig_loss1=sig_loss1_value)
#         #metric_logger.update(sig_loss2=sig_loss2_value)
#         #metric_logger.update(sig_loss3=sig_loss3_value)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])

#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# # corel5k
def train_one_epoch(model: torch.nn.Module, criterion: TPLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, scheduler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets_list = []
        base_target = []
        loss1_target = []
        for i in range(len(targets)):
            base_target.append(targets[i][:260])
            loss1_target.append((targets[i][260:]))
        for i in range(len(base_target)):
            base_target[i] = base_target[i].type(torch.float64).to(device, non_blocking=True)
        for i in range(len(loss1_target)):
            loss1_target[i] = loss1_target[i].type(torch.float64).to(device, non_blocking=True)
        targets_list.append(base_target)
        targets_list.append(loss1_target)
        with torch.cuda.amp.autocast():
            outputs_list = model(samples, targets_list)
            # base_loss,, targets_list,,,sig_loss0,sig_loss1,sig_loss4,sig_loss5,sig_loss6,sig_loss7,sig_loss8,sig_loss9,sig_loss10,sig_loss11,sig_loss2,sig_loss3,
            loss, sig_loss0, sig_loss1, base_loss = criterion(outputs_list, targets_list)
        base_loss_value = base_loss.item()
        sig_loss0_value = sig_loss0.item()
        sig_loss1_value = sig_loss1.item()
        # sig_loss2_value = sig_loss2.item()
        # sig_loss3_value = sig_loss3.item()
        # sig_loss4_value = sig_loss4.item()
        # sig_loss5_value = sig_loss5.item()
        # sig_loss6_value = sig_loss6.item()
        # sig_loss7_value = sig_loss7.item()
        # sig_loss8_value = sig_loss8.item()
        # sig_loss9_value = sig_loss9.item()
        # sig_loss10_value = sig_loss10.item()
        # sig_loss11_value = sig_loss11.item()
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        scheduler.step()
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        metric_logger.update(loss=loss_value)
        metric_logger.update(base_loss=base_loss_value)
        metric_logger.update(sig_loss0=sig_loss0_value)
        metric_logger.update(sig_loss1=sig_loss1_value)
        # metric_logger.update(sig_loss2=sig_loss2_value)
        # metric_logger.update(sig_loss3=sig_loss3_value)
        # metric_logger.update(sig_loss4=sig_loss4_value)
        # metric_logger.update(sig_loss5=sig_loss5_value)
        # metric_logger.update(sig_loss6=sig_loss6_value)
        # metric_logger.update(sig_loss7=sig_loss7_value)
        # metric_logger.update(sig_loss8=sig_loss8_value)
        # metric_logger.update(sig_loss9=sig_loss9_value)
        # metric_logger.update(sig_loss10=sig_loss10_value)
        # metric_logger.update(sig_loss11=sig_loss11_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# def train_one_epoch(model: torch.nn.Module, criterion: TPLoss,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler,scheduler, max_norm: float = 0,
#                     model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
#                     set_training_mode=True, args=None):

#     model.train(set_training_mode)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 100
#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
#         samples = samples.to(device, non_blocking=True)
#         with torch.cuda.amp.autocast():
#             outputs_list = model(samples,targets)
#             #base_loss,, targets_list,,,sig_loss0,sig_loss1,sig_loss3,sig_loss4,sig_loss5,sig_loss6,sig_loss7,sig_loss8,sig_loss9,sig_loss10,sig_loss11,sig_loss1,sig_loss0,
#             loss,base_loss= criterion(outputs_list, targets)
#         base_loss_value = base_loss.item()
#         #sig_loss0_value = sig_loss0.item()
#         #sig_loss1_value = sig_loss1.item()
#         #sig_loss2_value = sig_loss2.item()  sig_loss2,
#         # sig_loss3_value = sig_loss3.item()
#         # sig_loss4_value = sig_loss4.item()
#         # sig_loss5_value = sig_loss5.item()
#         # sig_loss6_value = sig_loss6.item()
#         # sig_loss7_value = sig_loss7.item()
#         # sig_loss8_value = sig_loss8.item()
#         # sig_loss9_value = sig_loss9.item()
#         # sig_loss10_value = sig_loss10.item()
#         # sig_loss11_value = sig_loss11.item()
#         loss_value = loss.item()
#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             sys.exit(1)
#         optimizer.zero_grad()
#         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#         loss_scaler(loss, optimizer, clip_grad=max_norm,
#                     parameters=model.parameters(), create_graph=is_second_order)
#         scheduler.step()
#         torch.cuda.synchronize()
#         if model_ema is not None:
#             model_ema.update(model)
#         metric_logger.update(loss=loss_value)
#         metric_logger.update(base_loss=base_loss_value)
#         #metric_logger.update(sig_loss0=sig_loss0_value)
#         #metric_logger.update(sig_loss1=sig_loss1_value)
#         #metric_logger.update(sig_loss2=sig_loss2_value)
#         # metric_logger.update(sig_loss3=sig_loss3_value)
#         # metric_logger.update(sig_loss4=sig_loss4_value)
#         # metric_logger.update(sig_loss5=sig_loss5_value)
#         # metric_logger.update(sig_loss6=sig_loss6_value)
#         # metric_logger.update(sig_loss7=sig_loss7_value)
#         # metric_logger.update(sig_loss8=sig_loss8_value)
#         # metric_logger.update(sig_loss9=sig_loss9_value)
#         # metric_logger.update(sig_loss10=sig_loss10_value)
#         # metric_logger.update(sig_loss11=sig_loss11_value)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# nus-wide
# def train_one_epoch(model: torch.nn.Module,criterion: TPLoss,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler,scheduler, max_norm: float = 0,
#                     model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
#                     set_training_mode=True, args=None):

#     model.train(set_training_mode)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 300

#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
#         samples = samples.to(device, non_blocking=True)
#         with torch.cuda.amp.autocast():
#             outputs_list = model(samples,targets)
#             base_loss,sig_loss_0,loss= criterion(outputs_list,targets)
#         loss_value = loss.item()
#         base_loss_value = base_loss.item()
#         sig_loss_0_value = sig_loss_0.item()
#         #sig_loss_1_value = sig_loss_1.item()sig_loss_1,

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             sys.exit(1)
#         optimizer.zero_grad()
#         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#         loss_scaler(loss, optimizer, clip_grad=max_norm,
#                     parameters=model.parameters(), create_graph=is_second_order)
#         scheduler.step()
#         torch.cuda.synchronize()
#         if model_ema is not None:
#             model_ema.update(model)
#         metric_logger.update(loss=loss_value)
#         metric_logger.update(base_loss=base_loss_value)
#         metric_logger.update(sig_loss_0=sig_loss_0_value)
#         #metric_logger.update(sig_loss_1=sig_loss_1_value)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, '_')
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
