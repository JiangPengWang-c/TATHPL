1  # Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

__all__ = [
    'tit_small_topic_patch16_224', 'tit_large_topic_patch16_224', 'tit_large_topic_patch16_448',
    'tit_base_topic_patch16_224', 'tit_base_topic_patch16_384',
    'tit_base_topic_patch16_448', 'tit_large_topic_patch32_384',
]



class TopicVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))  # self.embed_dim=768 or 384
        self.prototype_0 = nn.Linear(self.embed_dim, 1, bias=False)
        self.prototype_1 = nn.Linear(self.embed_dim, 1, bias=False)
        # self.prototype_2 = nn.Linear(self.embed_dim, 1, bias = False)
        # self.prototype_3 = nn.Linear(self.embed_dim, 1, bias = False)
        # self.prototype_4 = nn.Linear(self.embed_dim, 1, bias = False)#Three Topics + other
        # self.prototype_5 = nn.Linear(self.embed_dim, 1, bias = False)
        # self.prototype_6 = nn.Linear(self.embed_dim, 1, bias = False)
        # self.prototype_7 = nn.Linear(self.embed_dim, 1, bias = False)
        # self.prototype_8 = nn.Linear(self.embed_dim, 1, bias = False)
        # self.prototype_9 = nn.Linear(self.embed_dim, 1, bias = False)
        # self.prototype_10 = nn.Linear(self.embed_dim, 1, bias = False)
        # self.prototype_11 = nn.Linear(self.embed_dim, 1, bias = False)
        # self.prototype_12 = nn.Linear(self.embed_dim, 1, bias = False)

        trunc_normal_(self.pos_embed, std=.02)
        self.prototype_0.apply(self._init_weights)
        self.prototype_1.apply(self._init_weights)
        # self.prototype_2.apply(self._init_weights)
        # self.prototype_3.apply(self._init_weights)
        # self.prototype_5.apply(self._init_weights)
        # self.prototype_6.apply(self._init_weights)
        # self.prototype_7.apply(self._init_weights)
        # self.prototype_8.apply(self._init_weights)
        # self.prototype_9.apply(self._init_weights)
        # self.prototype_10.apply(self._init_weights)
        # self.prototype_11.apply(self._init_weights)
        # self.prototype_12.apply(self._init_weights)

        self.matrix_0 = nn.Linear(self.embed_dim, 2, bias=False)
        self.matrix_1 = nn.Linear(self.embed_dim, 3, bias=False)
        # self.matrix_2 = nn.Linear(self.embed_dim, 4, bias = False)
        # self.matrix_3 = nn.Linear(self.embed_dim, 5, bias = False)
        # self.matrix_4 = nn.Linear(self.embed_dim, 10, bias = False)
        # self.matrix_5 = nn.Linear(self.embed_dim, 15, bias = False)
        # self.matrix_6 = nn.Linear(self.embed_dim, 20, bias = False)
        # self.matrix_7 = nn.Linear(self.embed_dim, 25, bias = False)
        # self.matrix_8 = nn.Linear(self.embed_dim, 30, bias = False)
        # self.matrix_9 = nn.Linear(self.embed_dim, 35, bias = False)
        # self.matrix_10 = nn.Linear(self.embed_dim, 40, bias = False)
        # self.matrix_11 = nn.Linear(self.embed_dim, 50, bias = False)

        self.matrix_0.apply(self._init_weights)
        self.matrix_1.apply(self._init_weights)
        # self.matrix_2.apply(self._init_weights)
        # self.matrix_3.apply(self._init_weights)
        # self.matrix_4.apply(self._init_weights)
        # self.matrix_5.apply(self._init_weights)
        # self.matrix_6.apply(self._init_weights)
        # self.matrix_7.apply(self._init_weights)
        # self.matrix_8.apply(self._init_weights)
        # self.matrix_9.apply(self._init_weights)
        # self.matrix_10.apply(self._init_weights)
        # self.matrix_11.apply(self._init_weights)

    def forward_features_train(self, x, targets_list):

        B = x.shape[0]
        x = self.patch_embed(x)
        num_patch = x.shape[1]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)  # (B,P*P+1,embed_dim)
        ori = x.shape[1]
        x = x + self.pos_embed  # 好神奇的加
        x = self.pos_drop(x)
        prototypes = [self.prototype_0,
                      self.prototype_1,
                      # self.prototype_2,
                      # self.prototype_3,
                      #                       # self.prototype_4,
                      #                       #self.prototype_5, self.prototype_6, self.prototype_7, \
                      #                       # self.prototype_8, self.prototype_9, self.prototype_10,self.prototype_11
                      ]

        matrices = [self.matrix_0,
                    self.matrix_1,
                    # self.matrix_2,
                    # self.matrix_3, \
                    # self.matrix_4, self.matrix_5, self.matrix_6, self.matrix_7, \
                    # self.matrix_8, self.matrix_9, self.matrix_10,self.matrix_11
                    ]
        self.head = self.head
        emb_dim = x.shape[-1]  # 768 or 384
        n = 0
        proto_cls_probs = []
        # tmp_tensor = prototypes[0].expand(B,-1,-1)
        # x_with_prototype = torch.cat((x, tmp_tensor), dim=1)

        for b in range(len(self.blocks)):
            # if b==0 or b==11:
            #     tmp_tensor = prototypes[n].weight.expand(B, -1, -1)
            #     n+=1
            #     x = torch.cat((x, tmp_tensor), dim=1)
            #     x = self.blocks[b](x)

            #             if b==11:
            #                 tmp_tensor = prototypes[n].weight.expand(B, -1, -1)
            #                 x_with_prototype = torch.cat((x, tmp_tensor), dim=1)
            #                 x_with_prototype = self.blocks[b](x_with_prototype)
            #                 x = x_with_prototype[:, :ori, :]
            #                 proto_cls_prob = x_with_prototype[:, ori:, :] @ (matrices[n].weight.T)

            #                 proto_cls_probs.append(torch.diagonal(proto_cls_prob, dim1=-2, dim2=-1))
            #                 n+=1
            if b == 0 or b == 11:
                tmp_tensor = prototypes[n].weight.expand(B, -1, -1)
                x_with_prototype = torch.cat((x, tmp_tensor), dim=1)
                # x_with_prototype+=tmp_tensor
                # x_with_prototype[:,0,:]+=tmp_tensor.squeeze(1)
                x_with_prototype = self.blocks[b](x_with_prototype)
                x = x_with_prototype[:, :ori, :]
                proto_cls_prob = matrices[n](x_with_prototype[:, ori:, :])
                proto_cls_probs.append(proto_cls_prob)
                # proto_cls_probs.append(torch.diagonal(proto_cls_prob, dim1=-2, dim2=-1))
                n += 1
            else:
                x = self.blocks[b](x)
        x = self.blocks[-1](x)
        x = self.norm(x)
        # for i in range(len(prototypes)):
        #     x[:,0]+=prototypes[i].weight.expand(B, -1, -1).squeeze(1)
        return x[:, 0], proto_cls_probs

    # #

    def forward_features_val(self, x, targets_list):

        B = x.shape[0]
        x = self.patch_embed(x)
        num_patch = x.shape[1]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)  # (B,P*P+1,embed_dim)
        ori = x.shape[1]
        x = x + self.pos_embed  # 好神奇的加
        x = self.pos_drop(x)

        prototypes = [self.prototype_0,
                      self.prototype_1,
                      # self.prototype_2,
                      # self.prototype_3,
                      # self.prototype_4,
                      # self.prototype_5, self.prototype_6, self.prototype_7, \
                      # self.prototype_8, self.prototype_9, self.prototype_10,self.prototype_11
                      ]

        matrices = [self.matrix_0,
                    self.matrix_1,
                    # self.matrix_2,
                    # self.matrix_3, \
                    # self.matrix_4, self.matrix_5, self.matrix_6, self.matrix_7, \
                    # self.matrix_8, self.matrix_9, self.matrix_10,self.matrix_11
                    ]
        self.head = self.head
        emb_dim = x.shape[-1]  # 768 or 384
        n = 0

        proto_cls_probs = []

        for b in range(len(self.blocks)):
            # if b==0 or b==11:
            #     tmp_tensor = prototypes[n].weight.expand(B, -1, -1)
            #     n+=1
            #     x = torch.cat((x, tmp_tensor), dim=1)
            #     x = self.blocks[b](x)
            # else:
            #     x = self.blocks[b](x)
            #             if b==11:
            #                 tmp_tensor = prototypes[n].weight.expand(B, -1, -1)
            #                 x_with_prototype = torch.cat((x, tmp_tensor), dim=1)
            #                 x_with_prototype = self.blocks[b](x_with_prototype)
            #                 x = x_with_prototype[:, :ori, :]

            #                 n+=1
            if b == 0 or b == 11:
                tmp_tensor = prototypes[n].weight.expand(B, -1, -1)

                x_with_prototype = torch.cat((x, tmp_tensor), dim=1)
                # x_with_prototype+=tmp_tensor

                x_with_prototype = self.blocks[b](x_with_prototype)
                x = x_with_prototype[:, :ori, :]
                n += 1
            else:
                x = self.blocks[b](x)
        x = self.blocks[-1](x)
        x = self.norm(x)
        # for i in range(len(prototypes)):
        #     x[:,0]+=prototypes[i].weight.expand(B, -1, -1).squeeze(1)
        return x[:, 0]

    def forward(self, x, targets_list):
        if not self.training:
            # during inference, return the last classifier predictions
            x_final = self.forward_features_val(x, targets_list)
            x_final = self.head(x_final)
            return x_final
        if self.training:
            x_final, proto_cls_probs = self.forward_features_train(x, targets_list)
            x_final = self.head(x_final)
            return x_final, proto_cls_probs







@register_model
def tit_small_topic_patch16_224(pretrained=False, **kwargs):
    model = TopicVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        print('loading model state dict now!')
        checkpoint = torch.hub.load_state_dict_from_url(
            # url="https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/vit_base_patch16_224_miil_21k.pth",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth",
            # url='gs://vit_models/imagenet21k',
            map_location="cpu", check_hash=True
        )
        checkpoint.pop('head.weight')
        checkpoint.pop('head.bias')
        model.load_state_dict(checkpoint, strict=False)
    return model


@register_model
def tit_large_topic_patch16_224(pretrained=False, **kwargs):
    model = TopicVisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        print('loading model state dict now!')
        checkpoint = torch.load('/home/featurize/data/vit_large_patch16_224/pytorch_model.bin', map_location='cpu')
        checkpoint.pop('head.weight')
        checkpoint.pop('head.bias')
        model.load_state_dict(checkpoint, strict=False)
    return model


@register_model
def tit_large_topic_patch16_448(pretrained=False, **kwargs):
    model = TopicVisionTransformer(
        img_size=448, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        print('loading model state dict now!')
        checkpoint = torch.load('/home/featurize/data/vit_large_patch16_384/pytorch_model.bin', map_location='cpu')
        checkpoint = adapt_weights(checkpoint, model)
        model.load_state_dict(checkpoint, strict=False)
    return model


@register_model
def tit_base_topic_patch16_224(pretrained=False, **kwargs):
    model = TopicVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        print('loading model state dict now!')
        checkpoint = torch.hub.load_state_dict_from_url(
            # url="https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/vit_base_patch16_224_miil_21k.pth",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth",
            # url='gs://vit_models/imagenet21k',
            map_location="cpu", check_hash=True
        )
        checkpoint.pop('head.weight')
        checkpoint.pop('head.bias')
        model.load_state_dict(checkpoint, strict=False)
    return model


@register_model
def tit_base_topic_patch16_384(pretrained=False, **kwargs):
    model = TopicVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        print('loading model state dict now!')
        checkpoint = torch.load('/home/featurize/data/vit-base-patch-16-384/pytorch_model.bin', map_location='cpu')
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     #url="https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/vit_base_patch16_224_miil_21k.pth",
        #     url="hf_hub:timm/vit_base_patch16_384.augreg_in21k_ft_in1k",
        #     #url='gs://vit_models/imagenet21k',
        #     map_location="cpu", check_hash=True
        # )
        checkpoint.pop('head.weight')
        checkpoint.pop('head.bias')
        model.load_state_dict(checkpoint, strict=False)
    return model


@register_model
def tit_base_topic_patch16_448(pretrained=False, **kwargs):
    model = TopicVisionTransformer(
        img_size=448, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load('/home/featurize/data/vit-base-patch-16-384/pytorch_model.bin', map_location='cpu')
        checkpoint = adapt_weights(checkpoint, model)
        model.load_state_dict(checkpoint, strict=False)
    return model


@register_model
def tit_large_topic_patch32_384(pretrained=False, **kwargs):
    model = TopicVisionTransformer(
        img_size=384, patch_size=32, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = {
        'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        'num_classes': 1000,
        'input_size': (3, 384, 384),
        'pool_size': None,
        'crop_pct': 1.0,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        'architecture': 'deit_large_topic_patch32_384'
    }
    if pretrained:
        print('loading model state dict now!')
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint.pop('head.weight')
        checkpoint.pop('head.bias')
        model.load_state_dict(checkpoint, strict=False)
    return model


@register_model


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print('Resized position embedding: {} to {}'.format(posemb.shape, posemb_new.shape))
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    print('Position embedding grid-size from {} to {}'.format([gs_old, gs_old], gs_new))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def adapt_weights(checkpoint, model):
    # 去除classifiication head
    checkpoint.pop('head.weight')
    checkpoint.pop('head.bias')
    # 改变position encodeing
    pos_embed_w = checkpoint["pos_embed"]
    pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
        pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    checkpoint["pos_embed"] = pos_embed_w
    return checkpoint

