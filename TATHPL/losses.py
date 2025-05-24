# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the HiLoss, MaskSoftTargetCrossEntropy, knowledge distillation loss
"""
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import random


class TPLoss(torch.nn.Module):
    def __init__(self, base_criterion: torch.nn.Module, sig_criterion: torch.nn.Module, epoch):
        super().__init__()
        self.base_criterion = base_criterion
        self.sig_criterion = sig_criterion
        self.epoch = epoch

    #     def split_croase_label(self,targets):
    #         targets_list = []
    #         base_target = []
    #         loss1_target = []
    #         for i in range(len(targets)):
    #             base_target.append(targets[i][:20])
    #             loss1_target.append((targets[i][20:]))

    #         for i in range(len(base_target)):
    #             base_target[i] = base_target[i].type(torch.float64).cuda()
    #         for i in range(len(loss1_target)):
    #             loss1_target[i] = loss1_target[i].type(torch.float64).cuda()
    #         targets_list.append(base_target)
    #         targets_list.append(loss1_target)
    #        return targets_list
    #     def split_croase_label(self,targets):
    #         target=targets[0]
    #         topic=targets[1]

    #         target = target.cuda()
    #         target = target.max(dim=1)[0]
    #         target_list=[]
    #         target_list.append(target)
    #         target_list.append(topic)
    #         return target_list
    # nus:
    #     def split_croase_label(self,targets):
    #         target=targets[0]
    #         topic=targets[1]

    #         target = target.cuda()
    #         target_list=[]
    #         target_list.append(target)
    #         target_list.append(topic)
    #         return target_list
    ###voc2007:
    ##mse:
    #     def forward(self, outputs_list, labels_list):
    #         #labels_list = self.split_croase_label(labels_list)
    #         sof = nn.Softmax()

    #         sig_loss_0 = self.sig_criterion(sof(outputs_list[1][0].squeeze()).float(), torch.stack(labels_list[1],dim=1).float().cuda())
    #         #sig_loss_1 = self.sig_criterion(sof(outputs_list[1][1].squeeze()).float(), torch.stack(labels_list[2],dim=1).float().cuda())
    #         # sig_loss_2 = self.sig_criterion(sof(outputs_list[1][2].squeeze()).float(), torch.stack(labels_list[3],dim=1).float().cuda())

    #         base_label=labels_list[0].float().cuda()
    #         base_loss_final = self.base_criterion(outputs_list[0], base_label)
    #         weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    #         assert len(weights) >= 13
    #         loss = base_loss_final * weights[0]+ sig_loss_0 * weights[1]
    #         return loss,sig_loss_0,base_loss_final
    # + sig_loss_2 * weights[3]sig_loss_2,sig_loss_1,+ sig_loss_1 * weights[2]
    def forward(self, outputs_list, labels_list):
        # labels_list = self.split_croase_label(labels_list)

        croase_lable = torch.stack(labels_list[1])
        # print(outputs_list[1])
        sig_loss_0 = self.sig_criterion(outputs_list[1][0].squeeze(), croase_lable[:, 0].long())
        sig_loss_1 = self.sig_criterion(outputs_list[1][1].squeeze(), croase_lable[:, 1].long())
        # sig_loss_2 = self.sig_criterion(outputs_list[1][2].squeeze(), croase_lable[:,2].long())
        # sig_loss_3 = self.sig_criterion(outputs_list[1][3].squeeze(), croase_lable[:,3].long())
        # sig_loss_4 = self.sig_criterion(outputs_list[1][4].squeeze(), croase_lable[:,4].long())
        # sig_loss_5 = self.sig_criterion(outputs_list[1][5].squeeze(), croase_lable[:,5].long())
        # sig_loss_6 = self.sig_criterion(outputs_list[1][6].squeeze(), croase_lable[:,6].long())
        # sig_loss_7 = self.sig_criterion(outputs_list[1][7].squeeze(), croase_lable[:,7].long())
        # sig_loss_8 = self.sig_criterion(outputs_list[1][8].squeeze(), croase_lable[:,8].long())
        # sig_loss_9 = self.sig_criterion(outputs_list[1][9].squeeze(), croase_lable[:,9].long())
        # sig_loss_10 = self.sig_criterion(outputs_list[1][10].squeeze(), croase_lable[:,10].long())
        # sig_loss_11 = self.sig_criterion(outputs_list[1][11].squeeze(), croase_lable[:,11].long())

        base_label = torch.stack(labels_list[0])
        base_loss_final = self.base_criterion(outputs_list[0], base_label)
        weights = [1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert len(weights) >= 13
        # if self.flag:
        #     loss=sig_loss_0 * weights[1]
        # else:
        loss = base_loss_final * weights[0] + sig_loss_0 * weights[1] + sig_loss_1 * weights[2]
        # + sig_loss_4 * weights[5]+ sig_loss_5 * weights[6]+ sig_loss_6 * weights[7]+ sig_loss+ sig_loss_2 * weights[3]+ sig_loss_3 * weights[4]_7 * weights[8]+ sig_loss_8 * weights[9]+ sig_loss_9 * weights[10]+ sig_loss_10 * weights[11]+ sig_loss_11 * weights[12]
        # sig_loss_4,sig_loss_5,sig_loss_6,sig_loss_7,sig_loss_8,sig_loss_8,sig_loss_10,sig_loss_11,,sig_loss_2,sig_loss_3
        return loss, sig_loss_0, sig_loss_1, base_loss_final


##nus
#     def forward(self, outputs_list, labels_list):
#         labels_list = self.split_croase_label(labels_list)
#         croase_label=torch.stack(labels_list[1])
#         sig_loss_0 = self.sig_criterion(outputs_list[1][0].squeeze(), croase_label[0].long().cuda())
#         #sig_loss_1 = self.sig_criterion(outputs_list[1][1].squeeze(), croase_label[1].long().cuda())
#         base_label=labels_list[0]
#         base_loss_final = self.base_criterion(outputs_list[0], base_label)
#         weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#         assert len(weights) >= 13
#         loss = base_loss_final * weights[0]+ sig_loss_0 * weights[1]
# #+ sig_loss_1 * weights[2]sig_loss_1,
#         return base_loss_final,sig_loss_0,loss

#      ###mscoco：
#     def forward(self, outputs_list, labels_list):
#         labels_list = self.split_croase_label(labels_list)

#         croase_label=torch.stack(labels_list[1])
#         # print(croase_label)
#         # print(outputs_list[1][0].squeeze())
#         sig_loss_0 = self.sig_criterion(outputs_list[1][0].squeeze(), croase_label[0].long().cuda())
#         sig_loss_1 = self.sig_criterion(outputs_list[1][1].squeeze(), croase_label[1].long().cuda())
#         #sig_loss_2 = self.sig_criterion(outputs_list[1][2].squeeze(), croase_label[2].long().cuda())
#         #sig_loss_3 = self.sig_criterion(outputs_list[1][3].squeeze(), croase_label[3].long().cuda())
#         base_label = labels_list[0]
#         base_loss_final = self.base_criterion(outputs_list[0], base_label)
#         weights = [1,0.3,0.3,0.1,0.1,1,1,1,1,1,1,1,1,1,1,1]
#         assert len(weights) >= 13
#         loss = base_loss_final * weights[0]+ sig_loss_0 * weights[1]+ sig_loss_1 * weights[2]
# #+ sig_loss_3 * weights[4]sig_loss_3,+ sig_loss_2 * weights[3]sig_loss_2,
#         return base_loss_final,sig_loss_0,sig_loss_1,loss
#     def forward(self, outputs_list, labels_list):
#         labels_list = self.split_croase_label(labels_list)
#         sof = nn.Softmax()
#         croase_lable=torch.stack(labels_list[1]).transpose(0, 1)
#         sig_loss_0 = self.sig_criterion(sof(outputs_list[1][0].squeeze()).float().cuda(), croase_lable.float().cuda())

#         base_label=labels_list[0]
#         base_loss_final = self.base_criterion(outputs_list[0], base_label)
#         weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#         assert len(weights) >= 13
#         loss = base_loss_final * weights[0]+ sig_loss_0 * weights[1]
#         return base_loss_final,sig_loss_0,loss





class SigSoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SigSoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * torch.log(x), dim=-1)
        return loss.mean()


class MaskSoftTargetCrossEntropy(nn.Module):
    """
    This module wraps TATHPL standard SoftTargetCrossEntropy and adds an extra mask to filter out meaningless logits.
    """

    def __init__(self):
        super(MaskSoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: B x nb_classes, target: B x nb_classes, mask: B with 0 or 1
        """
        x = x[mask == 1]
        target = target[mask == 1]
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()



