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
import timm
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
#from randaugment import RandAugment
import os
import torchvision.transforms as transforms
from loss import AsymmetricLoss
from torch.cuda.amp import GradScaler, autocast
import utils
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    # Model parameters
    parser.add_argument('--model', default='tit_base_topic_patch16_224', type=str, metavar='MODEL',#deit_base_patch16_224
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin-mem', action='store_false',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=False)

    parser.add_argument('--data_corel5k', default="../Corel5k", type=str,
                        help='dataset path')
    parser.add_argument('--corel5k_num_class', default=260, type=int,
                        help='dataset path')

    
    return parser


def main(args):
    print(os.getcwd())
    utils.init_distributed_mode(args)

    print(args)


    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    


 
###corel5k:
    data_path=f'{args.data_corel5k}/Corel5k'
    lable_path_train= f'{args.data_corel5k}/Corel5k/total_train.txt'
    lable_path_test = f'{args.data_corel5k}/Corel5k/total_test.txt'
    
    test_dataset = Corel5k(data_path, lable_path_test, transforms.Compose([
        transforms.Resize((args.input_size,args.input_size)),
        transforms.ToTensor(),
    ]), None, False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True)
    total_dataset=Corel5k(data_path, lable_path_train, transforms.Compose([
                                       transforms.Resize((args.input_size, args.input_size)),
                                       CutoutPIL(cutout_factor=0.5),
                                       #RandAugment(),
                                       transforms.ToTensor(),
                                   ]),None,True)
    total_loader  = torch.utils.data.DataLoader(
        total_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.corel5k_num_class,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
    )
    print('successfully created')
   
    model.to(device)

    train_multi_label_coco(model, total_loader, test_loader, args.lr)

def train_multi_label_coco(model, train_loader, val_loader, lr):
    patience = 0

    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82
    output_dir = Path(args.output_dir)
    # set optimizer
    Epochs = 50
    weight_decay = 1e-4
    # criterion = TPLoss(nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss())

    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)
    highest_mAP = 0
    best_epoch = 0
    loss_scaler = NativeScaler()
    mAP=[]
    topic='2  after cls onecycle('+str(Epochs)+') lr='+str(lr)
    print(topic)
    for epoch in range(Epochs):
        criterion = TPLoss(AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True), nn.CrossEntropyLoss(reduction='mean'),epoch)
        train_stats = train_one_epoch(
            model, criterion, train_loader,
            optimizer, args.device, epoch, loss_scaler,scheduler,
            args.clip_grad, ema, None,
            set_training_mode=True,  # keep in eval mode during finetuning
            args = args,
        )
        model.eval()
        mAP_score= validate_multi(val_loader, model, ema,highest_mAP)
        mAP.append(mAP_score)
        model.train()

        if (mAP_score <= highest_mAP):patience+=1
        else :patience=0
        print(patience)
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            best_epoch=epoch

        print('current_mAP = {:.3f}, highest_mAP = {:.3f}\n'.format(mAP_score, highest_mAP))

        
def validate_multi(val_loader, model, ema_model,highest_mAP):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    preds_base = []
    targets_base=[]
    targets=[]
    for i, (samples, target) in enumerate(val_loader):
        targets_list = []
        base_target = []
        loss1_target = []
        for i in range(len(target)):
            base_target.append(target[i][:260])
            loss1_target.append((target[i][260:]))

        for i in range(len(base_target)):
            base_target[i] = base_target[i].type(torch.float64).cuda()
        for i in range(len(loss1_target)):
            loss1_target[i] = loss1_target[i].type(torch.float64).cuda()
        targets_list.append(base_target)
        targets_list.append(loss1_target)
        # print(targets_list)
        # compute output,target,target
        with torch.no_grad():
            with autocast():
                output_regular = model(samples.cuda(),target)
                output_regular=Sig(output_regular).cpu()
                output_ema = Sig(ema_model.module(samples.cuda(),target)).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        preds_base.append(output_regular.cuda())
        tmp=[]
        for j in range(len(target)):
            tmp.append(target[j][:260])
        targets_base.append(torch.stack(tmp).cuda())
        targets.append(torch.stack(tmp).cpu().detach())
        # targets_base.append(target[0].cuda())
        # targets.append(target[0].cpu().detach())
        
    
    a=torch.cat(preds_base)

    preds_binary = torch.where(a > 0.7, torch.tensor(1), torch.tensor(0)).cpu()

    Hamming_loss = hamming_loss(torch.cat(preds_base),torch.cat(targets_base))
    mcm=multilabel_confusion_matrix(torch.cat(targets).numpy(),preds_binary)
    mif1 = micro_f1(mcm)
    maf1 = macro_f1(mcm)
    one_err=one_error(torch.cat(targets).numpy(),torch.cat(preds_base))
    auc = get_auc(torch.cat(targets).numpy(),torch.cat(preds_regular).numpy())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.3f}, mAP score EMA {:.3f},OF1 score{:.3f},OP score{:.3f},OR1 score{:.3f},CF1 score{:.3f},CP score{:.3f},CR score{:.3f},one_error score {:.6f},auc {:.6f}, Hamming_Loss score{:.6f},".format(mAP_score_regular, mAP_score_ema,mif1[0],mif1[1],mif1[2],maf1[0],maf1[1],maf1[2],one_err,auc,Hamming_loss,))

    return max(mAP_score_regular,mAP_score_ema)

from sklearn.metrics import hamming_loss

def computer_hamming_loss(y_true,y_pred):
    y_hot = torch.round(y_pred)
    #y_true = y_true.max(dim=1)[0].type(torch.float64)
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

    for i in range(targets.shape[0]):
        H = f1_score(targets[i, :].cpu(), binary_preds[i, :].cpu())
        f1.append(H)

    return np.mean(f1)
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
