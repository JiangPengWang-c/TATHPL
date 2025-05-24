# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,zero_one_loss,multilabel_confusion_matrix,label_ranking_loss,roc_auc_score
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import  MaskSoftTargetCrossEntropy, SigSoftTargetCrossEntropy,TPLoss
from samplers import RASampler
from torch.optim import lr_scheduler
from torch.utils.data import random_split
import model_learn

from helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, \
    add_weight_decay,Corel5k,compute_mAP,micro_f1,macro_f1,one_error,get_auc
from randaugment import RandAugment

import os
import torchvision.transforms as transforms
from loss import AsymmetricLoss
from torch.cuda.amp import GradScaler, autocast
#from randaugment import RandAugment

import utils



import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)

    parser.add_argument('--model', default='tit_base_topic_patch16_224', type=str, metavar='MODEL',#deit_base_patch16_224
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.set_defaults(pin_mem=False)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--data_coco', default="/home/featurize/data//coco2014", type=str,
                        help='dataset path')

    parser.add_argument('--coco_num_class', default=80, type=int,
                        help='dataset path')
    parser.add_argument('--data_corel5k', default="/home/featurize/data/Corel5k", type=str,
                        help='dataset path')
    parser.add_argument('--corel5k_num_class', default=260, type=int,
                        help='dataset path')
    parser.add_argument('--data_voc2012', default="/home/featurize/data/VOCtrainval_11-May-2012", type=str,
                        help='dataset path')
    parser.add_argument('--voc2012_num_class', default=20, type=str,
                        help='dataset path')
    
    return parser


def main(args):
    print(os.getcwd())
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    



#MS COCO
    instances_path_val = '/home/featurize/data/coco2014/annotations/instances_val2014.json'
    instances_path_train =  '/home/featurize/data/coco2014/annotations/instances_train2014.json'
    data_path_val = f'{args.data_coco}/val2014'  # args.data
    data_path_train = f'{args.data_coco}/train2014'  # args.data
    test_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.input_size, args.input_size)),
                                    transforms.ToTensor()
                                ]),None,False)
    train_dataset = CocoDetection(data_path_train,
                                  instances_path_train,
                                  transforms.Compose([
                                      transforms.Resize((args.input_size, args.input_size)),
                                      CutoutPIL(cutout_factor=0.5),
                                      RandAugment(),
                                      transforms.ToTensor()
                                  ]),None,True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,drop_last=True)#coco数据集当batch size = 32时，会出现最后一个step中样本大小为1的情况，会导致CE计算出问题

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.coco_num_class,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    model = nn.DataParallel(model,device_ids=[i for i in range(1)])
    print('successfully created')
   
    model.to(device)

    train_multi_label_coco(model, train_loader,test_loader, args.lr)






def train_multi_label_coco(model, train_loader, val_loader, lr):
    patience=0
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82s

    # set optimizerfilter(lambda p: p.requires_grad, model.parameters())
    Epochs = 50
    weight_decay = 1e-4
    #criterion = TPLoss(AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True), nn.MSELoss(reduction='sum'),0)
    criterion = TPLoss(AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True), nn.CrossEntropyLoss(reduction='mean'),0)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0.0)  # true wd, filter_bias_and_bn
    # params_other = []
    # params_prototype = []
    # for name, param in model.named_parameters():
    #     if name in ['prototype_0','prototype_1','matrix_0','matrix_1']:  # 假设 prototype 层的名称中包含 'prototype'
    #         params_prototype.append(param)
    #     else:
    #         params_other.append(param)
    # optimizer = torch.optim.Adam([
    #         {"params":params_other,"lr":lr},
    #         {"params":params_prototype,"lr":1e-2}
    #         ], lr=lr, weight_decay=0)# true wd, filter_bias_and_bn 
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,pct_start=0.2)

    highest_mAP = 0
    best_epoch=0
    trainInfoList = []
    loss_scaler = NativeScaler()
    topic='3 after cls'
    for epoch in range(Epochs):
        train_stats = train_one_epoch(
            model,criterion, train_loader,
            optimizer, args.device, epoch, loss_scaler, scheduler, 
            args.clip_grad, ema, None,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            args = args,
        )
        model.eval()    
        mAP_score,preds,targets = validate_multi(val_loader, model, ema,highest_mAP)
        model.train()
        if (mAP_score <= highest_mAP):patience+=1
        else :patience=0
        print(patience)
        if mAP_score > highest_mAP:
            if epoch>=7:
                res = {
                    'preds':preds,
                    'targets':targets
                }
                torch.save(res,'/home/featurize/work/coco_res/coco_res_20in6_384.pth')
            highest_mAP = mAP_score
            best_epoch=epoch
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))
        if(patience == 10):
            msg='highest_mAP = {:.3f} in epoch: {:.0f}'.format(highest_mAP,best_epoch)
            print(msg)
            with open('/home/featurize/work/coco_log.txt','TATHPL') as file:
                file.writelines("topic:"+topic+"  "+msg+'\n')

def validate_multi(val_loader, model, ema_model,highest_mAP):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    preds_base=[]
    steps_per_epoch = len(val_loader)
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0

    for i, (img, targetss) in enumerate(val_loader):
        target=targetss[0]
        topic=targetss[1]
        #target = target.cuda()
        target = target.max(dim=1)[0]
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(img.cuda(),targetss)).cpu()
                output_ema = Sig(ema_model.module(img.cuda(),targetss)).cpu()
        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())
        preds_base.append(output_regular.cuda())
    preds_binary = []
    a=torch.cat(preds_ema)
    preds_binary = torch.where(a > 0.8, torch.tensor(1), torch.tensor(0)).cpu()    
    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    Hamming_loss = hamming_loss(torch.cat(preds_ema), torch.cat(targets).cuda())
    mcm = multilabel_confusion_matrix(torch.cat(targets).numpy(), preds_binary)
    mif1 = micro_f1(mcm)
    maf1 = macro_f1(mcm)
    one_err = one_error(torch.cat(targets).numpy(), torch.cat(preds_ema))
    auc = get_auc(torch.cat(targets).numpy(),torch.cat(preds_regular).numpy())
    print("mAP score regular {:.5f}, mAP score EMA {:.5f},OF1 score{:.3f},OP score{:.3f},OR1 score{:.3f},CF1 score{:.3f},CP score{:.3f},CR score{:.3f},one_error score {:.3f},auc {:.3f}, Hamming_Loss score{:.3f},".format(mAP_score_regular, mAP_score_ema,mif1[0],mif1[1],mif1[2],maf1[0],maf1[1],maf1[2],one_err,auc,Hamming_loss,))
    return max(mAP_score_regular, mAP_score_ema),preds_base,targets
 
from sklearn.metrics import hamming_loss

def computer_hamming_loss(y_true,y_pred):
    y_hot = torch.round(y_pred)
    HammingLoss = []
    for i in range(y_pred.shape[0]):
        H = hamming_loss(y_true[i, :], y_hot[i, :])
        HammingLoss.append(H)
    res=torch.stack(HammingLoss)
    return torch.mean(res)
def hamming_loss(preds, targets):
    """
    计算多标签分类Hamming Loss的函数。
    :param preds: 预测的概率值，大小为 [batch_size, num_classes]
    :param targets: 目标标签值，大小为 [batch_size, num_classes]
    :return: 多标签分类Hamming Loss的值，大小为 [1]
    """
    # 将概率值转换为二进制标签（0或1）

    binary_preds = torch.round(preds)
    # 计算Hamming Loss
    hamming_loss = 1 - (binary_preds == targets).float().mean()
    return hamming_loss
def draw_mAP(mAP):

    x_axis_data=(np.ones(len(mAP)))
    x_axis_data=np.cumsum((x_axis_data))
    plt.plot(x_axis_data, mAP, 'b*--', alpha=0.5, linewidth=1, label='mAP')  # 'bo-'表示蓝色实线，数据点实心原点标注
    ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
    plt.legend()  # 显示上面的label
    plt.xlabel('Epoch')  # x_label
    plt.ylabel('mAP')  # y_label

    plt.show()
def f1_loss(preds, targets):
    binary_preds = torch.round(preds)
    f1=[]
    one_t=0
    one_p=0
    for i in range(len(binary_preds[0])):
        if binary_preds[0][i]==1:
            one_t=one_t+1
    #print(one_t)
    # print(precision_score(targets[0, :].cpu(), binary_preds[0, :].cpu()))
    # print(recall_score(targets[0, :].cpu(), binary_preds[0, :].cpu()))
    # print(f1_score(targets[0, :].cpu(), binary_preds[0, :].cpu()))
    for i in range(targets.shape[0]):
        H = f1_score(targets[i, :].cpu(), binary_preds[i, :].cpu())
        f1.append(H)
    # res = torch.stack(f1)
    return np.mean(f1)
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
