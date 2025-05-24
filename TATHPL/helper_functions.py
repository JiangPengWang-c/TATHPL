import os
from copy import deepcopy
import random
import time
from copy import deepcopy

import numpy as np
from PIL import Image
from torchvision import datasets as datasets
import torch
from PIL import ImageDraw
from pycocotools.coco import COCO
import json
import torch.utils.data as data
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import pickle
from sklearn.metrics import average_precision_score,roc_auc_score
 
def get_auc(target,preds):
    total_auc=0.
    auc=0
    for i in range(target.shape[1]):
        try:
            auc = roc_auc_score(target[:, i], preds[:, i])
        except ValueError:
            pass
        total_auc += auc

    multi_auc = total_auc / target.shape[1]

    return multi_auc

def one_error(target,preds):
    ins_num=preds.shape[0]
    class_num=preds.shape[1]
    err=0
    for i in range(ins_num):
        idx=torch.argmax(preds[i])
        if target[i][idx]==0:
            err+=1
    return err/ins_num


def micro_f1(mcm):
    class_num=mcm.shape[0]
    tp=0
    fp=0
    tn=0
    fn=0
    for i in range(class_num):
        tn=tn+mcm[i][0][0]
        fp=fp+mcm[i][0][1]
        fn=fn+mcm[i][1][0]
        tp=tp+mcm[i][1][1]
    OP=tp/(fp+tp)
    OR=tp/(tp+fn)
    OF1=(2*OP*OR)/(OP+OR)
    print("OP,OR,OF1",OP,OR,OF1)
    return OF1,OP,OR
def macro_f1(mcm):
    class_num=mcm.shape[0]

    CP=0
    CR=0
    for i in range(class_num):
        if mcm[i][0][1]+mcm[i][1][1] == 0:
            CP=CP
        else :
            CP=CP+(mcm[i][1][1]/(mcm[i][0][1]+mcm[i][1][1]))
        if mcm[i][1][0]+mcm[i][1][1] == 0:
            CR=CR
        else:
            CR=CR+(mcm[i][1][1]/(mcm[i][1][0]+mcm[i][1][1]))

   
    CP=CP/class_num
    CR=CR/class_num
    CF1=(2*CR*CP)/(CR+CP)
    print("CP,CR,CF1",CP,CR,CF1)
    return CF1,CP,CR


def compute_mAP(y_true, y_pred):
    AP = []
    # y_true = y_true.max(axis=1).astype(np.float64)
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)*100
def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds,write=False):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))


    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)

    return 100 * ap.mean()

class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01

class voc2007(torch.utils.data.Dataset):
    def __init__(self, data_path, lable_path, transform=None, target_transform=None, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.data_path = data_path
        self.lable_path = lable_path
        self.img_files = os.listdir(self.data_path)
        self.id2labels = {}
        with open(self.lable_path, 'rb') as pkl_file:
            self.id2labels = pickle.load(pkl_file)
        self.transform = transform
        self.target_transform = target_transform 
        self.topic_dict = {}
        if is_train:
            file = open("/home/featurize/data/voc2007/train/img_to_index5+6(300).txt") 
        else:
            file = open("/home/featurize/data/voc2007/train/img_to_index5(300).txt")
        total_topics = file.readlines()
        for i in range(len(total_topics)):
            res = total_topics[i].strip().split(':')
            res_topic = res[1].split(' ')
            self.topic_dict[res[0]] = res_topic
        
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, index):
        
        img_file = self.img_files[index]
        img_path = os.path.join(self.data_path, img_file)
        image = Image.open(img_path).convert('RGB')
        id = img_file.strip().split('.')[0]
        target = self.id2labels[id+'.xml']
        if self.is_train:
            topics = self.topic_dict[id]
        else:topics=[]
        topic = [int(item) for item in topics]
        target = target + topic  # 根据是否添加粗类决定要不要注释
        target = torch.Tensor(target)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class nus(torch.utils.data.Dataset):
    def __init__(self, data_path, lable_path, transform=None, target_transform=None, is_train=True):
        super().__init__()
        self.base_dir='NUS-WIDE/Flickr/'
        self.is_train = is_train 
        self.data_path = data_path
        self.lable_path = lable_path

        a = open(data_path)
        lines = a.readlines()
        self.img_id = []
        for line in lines:
            line = line.strip().split('Flickr')[1][1:]
            self.img_id.append(line)
            
        b=open(lable_path)
        lines=b.readlines()
        self.lables=[]
        for line in lines:
            line = line.strip().split(' ')
            self.lables.append(line)
            
        self.transform = transform
        self.target_transform = target_transform
        self.topic_dict = {}
        if is_train:
            file = open("NUS-WIDE/mine/train/img_to_index2(300).txt")
            total_topics = file.readlines()
            for i in range(len(total_topics)):
                res = total_topics[i].strip().split(':')
                res_topic = res[1].split(' ') 
                res_topic=list(map(float, res_topic)) 
                self.topic_dict[res[0]] = res_topic
        else:
            file = open("NUS-WIDE/mine/val/res50/result2/result2+epoch0.txt")
            total_topics = file.readlines()
            for i in range(len(total_topics)):
                res = total_topics[i].strip().split(' ')
                res_topic = res[1].split(' ') 
                res_topic=list(map(float, res_topic))
                self.topic_dict[res[0]] = res_topic

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, index):
        img_id=self.img_id[index]
        img_path=self.base_dir+img_id
        image = Image.open(img_path).convert('RGB')
        target = torch.Tensor(list(map(float, self.lables[index])))
        # print(target)
        # print(self.topic_dict[img_id])
        topic = self.topic_dict[img_id]
        
       
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, [target,topic]


class Corel5k(torch.utils.data.Dataset):
    def __init__(self,data_path,lable_path,transform=None,target_transform=None,is_train=True):
        super().__init__()
        self.is_train=is_train
        self.data_path=data_path
        self.lable_path=lable_path

        a=open(lable_path)
        lines=a.readlines()
        self.lables=[]
        for line in lines:
            line = line.strip().split(' ')
            self.lables.append(line)
        self.transform = transform
        self.target_transform = target_transform
        self.topic_dict = {} 
        if is_train:
            file = open("../Corel5k/target/train/img_to_index2+3(300).txt")
            total_topics = file.readlines()
            for i in range(len(total_topics)):
                res=total_topics[i].strip().split(':')
                res_topic=res[1].split(' ')
                self.topic_dict[res[0]]=res_topic  
            
        else:
            #The file is meaningless and only exists to ensure smooth code execution
            file = open("../Corel5k/target/val/val.txt")
            total_topics = file.readlines()
            for i in range(len(total_topics)):
                res=total_topics[i].strip().split(' ')
                res_topic=res[1].split(' ')
                self.topic_dict[res[0]]=res_topic  

    def __len__(self):
        return len(self.lables)
    def __getitem__(self, index):
        img_file=self.lables[index][0]
        img_id=self.lables[index][1]
        img_path= os.path.join(self.data_path+'/'+img_file+'/'+img_id+'.jpeg')
        image = Image.open(img_path).convert('RGB')
        target=self.lables[index][2:]
        topic=self.topic_dict[img_id]
        target=target+topic
        target = list(map(float, target))
        target = torch.Tensor(target)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image,target



class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None,is_train=False):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict() 
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        if is_train:
            file=open("MSCOCO/targets2014/train/img_to_index2+6(300).txt")
            topics=file.readlines()
            self.img_to_topic={}
            for i in range(len(topics)):
                res=topics[i].strip().split(':')
                res_topic = res[1].split(' ')
                res_topic_float = list(map(float,res_topic))
                self.img_to_topic[res[0]] = res_topic_float
                # self.img_to_topic[res[0]]=res_topic
        else:
            file=open("MSCOCO/targets2014/val/img_to_index2(300).txt")
            topics=file.readlines()
            self.img_to_topic={}
            for i in range(len(topics)):
                res=topics[i].strip().split(':')
                res_topic = res[1].split(' ')
                res_topic_float = list(map(float,res_topic))
                self.img_to_topic[res[0]] = res_topic_float
                # self.img_to_topic[res[0]]=res_topic
        

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        topic=self.img_to_topic[str(img_id)]
        return img, [target,topic]

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make TATHPL copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def get_class_ids_split(json_path, classes_dict):
    with open(json_path) as fp:
        split_dict = json.load(fp)
    if 'train class' in split_dict:
        only_test_classes = False
    else:
        only_test_classes = True

    train_cls_ids = set()
    val_cls_ids = set()
    test_cls_ids = set()

    # classes_dict = self.learn.dbunch.dataset.classes
    for idx, (i, current_class) in enumerate(classes_dict.items()):
        if only_test_classes:  # base the division only on test classes
            if current_class in split_dict['test class']:
                test_cls_ids.add(idx)
            else:
                val_cls_ids.add(idx)
                train_cls_ids.add(idx)
        else:  # per set classes are provided
            if current_class in split_dict['train class']:
                train_cls_ids.add(idx)
            # if current_class in split_dict['validation class']:
            #     val_cls_ids.add(i)
            if current_class in split_dict['test class']:
                test_cls_ids.add(idx)

    train_cls_ids = np.fromiter(train_cls_ids, np.int32)
    val_cls_ids = np.fromiter(val_cls_ids, np.int32)
    test_cls_ids = np.fromiter(test_cls_ids, np.int32)
    return train_cls_ids, val_cls_ids, test_cls_ids


def update_wordvecs(model, train_wordvecs=None, test_wordvecs=None):
    if hasattr(model, 'fc'):
        if train_wordvecs is not None:
            model.fc.decoder.query_embed = train_wordvecs.transpose(0, 1).cuda()
        else:
            model.fc.decoder.query_embed = test_wordvecs.transpose(0, 1).cuda()
    elif hasattr(model, 'head'):
        if train_wordvecs is not None:
            model.head.decoder.query_embed = train_wordvecs.transpose(0, 1).cuda()
        else:
            model.head.decoder.query_embed = test_wordvecs.transpose(0, 1).cuda()
    else:
        print("model is not suited for ml-decoder")
        exit(-1)


def default_loader(path):
    img = Image.open(path)
    return img.convert('RGB')
    # return Image.open(path).convert('RGB')

class DatasetFromList(data.Dataset):
    """From List dataset."""

    def __init__(self, root, impaths, labels, idx_to_class,
                 transform=None, target_transform=None, class_ids=None,
                 loader=default_loader):
        """
        Args:

            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on TATHPL sample.
        """
        self.root = root
        self.classes = idx_to_class
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.samples = tuple(zip(impaths, labels))
        self.class_ids = class_ids
        self.get_relevant_samples()

    def __getitem__(self, index):
        impath, target = self.samples[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform([target])
        target = self.get_targets_multi_label(np.array(target))
        if self.class_ids is not None:
            target = target[self.class_ids]
        return img, target

    def __len__(self):
        return len(self.samples)

    def get_targets_multi_label(self, target):
        # Full (non-partial) labels
        labels = np.zeros(len(self.classes))
        labels[target] = 1
        target = labels.astype('float32')
        return target

    def get_relevant_samples(self):
        new_samples = [s for s in
                       self.samples if any(x in self.class_ids for x in s[1])]
        # new_indices = [i for i, s in enumerate(self.samples) if any(x in self.class_ids for x
        #                                                             in s[1])]
        # omitted_samples = [s for s in
        #                    self.samples if not any(x in self.class_ids for x in s[1])]

        self.samples = new_samples



def parse_csv_data(dataset_local_path, metadata_local_path):
    try:
        df = pd.read_csv(os.path.join(metadata_local_path, "data.csv"))
    except FileNotFoundError:
        # No data.csv in metadata_path. Try dataset_local_path:
        metadata_local_path = dataset_local_path
        df = pd.read_csv(os.path.join(metadata_local_path, "data.csv"))
    images_path_list = df.values[:, 0]
    # images_path_list = [os.path.join(dataset_local_path, images_path_list[i]) for i in range(len(images_path_list))]
    labels = df.values[:, 1]
    image_labels_list = [labels.replace('[', "").replace(']', "").split(', ') for labels in
                             labels]

    if df.values.shape[1] == 3:  # split provided
        valid_idx = [i for i in range(len(df.values[:, 2])) if df.values[i, 2] == 'val']
        train_idx = [i for i in range(len(df.values[:, 2])) if df.values[i, 2] == 'train']
    else:
        valid_idx = None
        train_idx = None

    # logger.info("em: end parsr_csv_data: num_labeles: %d " % len(image_labels_list))
    # logger.info("em: end parsr_csv_data: : %d " % len(image_labels_list))

    return images_path_list, image_labels_list, train_idx, valid_idx


def multilabel2numeric(multilabels):
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(multilabels)
    classes = multilabel_binarizer.classes_
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    multilabels_numeric = []
    for multilabel in multilabels:
        labels = [class_to_idx[label] for label in multilabel]
        multilabels_numeric.append(labels)
    return multilabels_numeric, class_to_idx, idx_to_class


def get_datasets_from_csv(dataset_local_path, metadata_local_path, train_transform,
                          val_transform, json_path):

    images_path_list, image_labels_list, train_idx, valid_idx = parse_csv_data(dataset_local_path, metadata_local_path)
    labels, class_to_idx, idx_to_class = multilabel2numeric(image_labels_list)

    images_path_list_train = [images_path_list[idx] for idx in train_idx]
    image_labels_list_train = [labels[idx] for idx in train_idx]

    images_path_list_val = [images_path_list[idx] for idx in valid_idx]
    image_labels_list_val = [labels[idx] for idx in valid_idx]

    train_cls_ids, _, test_cls_ids = get_class_ids_split(json_path, idx_to_class)

    train_dl = DatasetFromList(dataset_local_path, images_path_list_train, image_labels_list_train,
                               idx_to_class,
                               transform=train_transform, class_ids=train_cls_ids)

    val_dl = DatasetFromList(dataset_local_path, images_path_list_val, image_labels_list_val, idx_to_class,
                             transform=val_transform, class_ids=test_cls_ids)

    return train_dl, val_dl, train_cls_ids, test_cls_ids
