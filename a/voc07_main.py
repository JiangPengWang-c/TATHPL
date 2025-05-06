# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import datetime
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
from losses import DistillationLoss, HiLoss, MaskSoftTargetCrossEntropy, SigSoftTargetCrossEntropy,TPLoss
from samplers import RASampler
from augment import new_data_aug_generator
from torch.optim import lr_scheduler
from torch.utils.data import random_split
import model_learn
import models_v2

from helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, \
    add_weight_decay,Corel5k,compute_mAP,micro_f1,macro_f1,one_error,get_auc,nus,voc2007
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
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')
 
    # Model parameters
    parser.add_argument('--model', default='tit_base_topic_patch16_448', type=str, metavar='MODEL',#deit_base_patch16_224
                        help='Name of model to train')
    parser.add_argument('--input-size', default=448, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--model-ema', action='store_true') 
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)') 
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)') 
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    parser.add_argument('--src', action='store_true') #simple random crop
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    # Dataset parameters
    parser.add_argument('--data-path', default="E:/download-web/imagenet", type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin-mem', action='store_false',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_true', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=False)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='tcp://localhost:11378', help='url used to set up distributed training')
    #parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--data_coco', default="/home/featurize/data/coco", type=str,
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
    parser.add_argument('--data_nus', default="/home/featurize/data/nus-wide", type=str,
                        help='dataset path')
    parser.add_argument('--nus_class', default=81, type=str,
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
     
#nus-wide:
    
    data_path_train = '/home/featurize/data/voc2007/trainval_images'
    data_path_test = '/home/featurize/data/voc2007/test_images'
    lable_path_train = '/home/featurize/data/voc2007/id2label.pkl'
    lable_path_test = '/home/featurize/data/voc2007/id2label.pkl'
    
    test_dataset = voc2007(data_path_test, lable_path_test, transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
    ]), None, False)#args.batch_size
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False,
        num_workers=0, pin_memory=True)
    total_dataset=voc2007(data_path_train,lable_path_train, transforms.Compose([
                                       transforms.Resize((args.input_size, args.input_size)),
                                       CutoutPIL(cutout_factor=0.5),
                                       RandAugment(),
                                       transforms.ToTensor(),
                                     #   normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     # std=[0.229, 0.224, 0.225]),
                                   ]),None,True)
    total_loader  = torch.utils.data.DataLoader(
        total_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    print(len(total_dataset))


    mixup_fn = None
    mixup_active = False #args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        '''
        nb_classes_list = [2, 2, 2, 4, 9, 16, 25, 49, 90, 170, 406, 1000]
        mixup_fns = []
        # There is a bug. class 0 is wrong
        for nb_classes in nb_classes_list:
            mixup_fns.append(
                Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=nb_classes)
            )
        '''
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
            
        

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.voc2012_num_class,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    # for name, param in model.named_parameters():
    #     print(f'Parameter {name} requires_grad: {param.requires_grad}')
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print('The network has {} params.'.format(params))
    # for name, param in model.named_parameters():
    #     # if 'prototype_0' not in name and 'matrix_0' not in name and 'head' not in name:
    #     if 'head' not in name:
    #         param.requires_grad = False
    # for name, param in model.named_parameters():
    #     print(f'Parameter {name} requires_grad: {param.requires_grad}')

    model = nn.DataParallel(model,device_ids=[i for i in range(3)])
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
    criterion = TPLoss(AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True), nn.CrossEntropyLoss(),0)
    #criterion = TPLoss(AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True), nn.MSELoss(reduction='sum'))
    #criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)filter(lambda p: p.requires_grad, model.parameters())
    parameters = add_weight_decay(model, weight_decay)

    
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
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)

    highest_mAP = 0
    best_epoch = 0
    #loss_scaler = GradScaler()
    loss_scaler = NativeScaler()
    topic='2 after cls one-cycle(50) lr = '+str(lr)

    for epoch in range(Epochs):
        train_stats = train_one_epoch(
            model,criterion, train_loader,
            optimizer, args.device, epoch, loss_scaler, scheduler,
            args.clip_grad, ema, None,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            args = args,
        )
        model.eval()
        mAP_score= validate_multi(val_loader, model, ema,highest_mAP)
        
        model.train()
        if (mAP_score <= highest_mAP):patience+=1
        else :patience=0
        print(patience)
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            best_epoch=epoch
        #     try:
        #         print('saving now!')
        #         torch.save(model.state_dict(), os.path.join(
        #         output_dir, 'model-highest.ckpt'))
        #     except:
        #         pass
        print('current_mAP = {:.3f}, highest_mAP = {:.3f}\n'.format(mAP_score, highest_mAP))
        if(patience == 10):
            msg='highest_mAP = {:.3f} in epoch: {:.0f}'.format(highest_mAP,best_epoch)
            print(msg)
            with open('/home/featurize/work/voc_log.txt','a') as file:
                file.writelines("topic:"+topic+"  "+msg+'\n')
            # break
#new one:
def validate_multi(val_loader, model, ema_model,highest_mAP):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    preds_base = []  
    val_loss = []
    targets_base=[]
    targets=[]
    for i, (samples, target) in enumerate(val_loader):          
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = model(samples.cuda(),target)
                output_regular=Sig(output_regular).cpu()
                output_ema = Sig(ema_model.module(samples.cuda(),target)).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        preds_binary=[]

        preds_base.append(output_regular.cuda())
        for i in range(len(preds_base)):
            for j in range(len(preds_base[i])):
                preds_binary.append(torch.round(preds_base[i][j]).cpu().numpy())
        tmp=[]
        for j in range(len(target)):
            tmp.append(target[j][:20])
        targets_base.append(torch.stack(tmp).cuda())
        targets.append(torch.stack(tmp).cpu().detach())
        # targets_base.append(target[0].cuda())
        # targets.append(target[0].cpu().detach())
    
    Hamming_loss = hamming_loss(torch.cat(preds_base),torch.cat(targets_base))
    mcm=multilabel_confusion_matrix(torch.cat(targets).numpy(),preds_binary)
    mif1 = micro_f1(mcm)
    maf1 = macro_f1(mcm)
    one_err=one_error(torch.cat(targets).numpy(),torch.cat(preds_base))
    auc = get_auc(torch.cat(targets).numpy(),torch.cat(preds_regular).numpy())
    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy(),True)
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    if highest_mAP<=max(mAP_score_regular,mAP_score_ema):
        torch.save(targets,'/home/featurize/work/re/voc_target_2.pth')
        torch.save(preds_regular,'/home/featurize/work/re/voc_preds_2.pth')
        torch.save(preds_ema,'/home/featurize/work/re/voc_ema_2.pth')
    print("mAP score regular {:.3f}, mAP score EMA {:.3f},OF1 score{:.3f},OP score{:.3f},OR1 score{:.3f},CF1 score{:.3f},CP score{:.3f},CR score{:.3f},one_error score {:.3f},auc {:.3f}, Hamming_Loss score{:.3f},".format(mAP_score_regular, mAP_score_ema,mif1[0],mif1[1],mif1[2],maf1[0],maf1[1],maf1[2],one_err,auc,Hamming_loss,))

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
