import os
import sys
import argparse
import numpy as np  # 导入 numpy


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

parser = argparse.ArgumentParser(description='PyTorch Training')
#parser.add_argument('--dataset', help='(coco,voc,nuswide)', default='coco')
parser.add_argument('--train_one_hot_vector_path', help='path to dataset', default="Corel5k/Corel5k/my_train_label.txt")
parser.add_argument('--output_path', help='', default="Corel5k/target/train/index/train_tp_class2(300).txt")
parser.add_argument('--n_features',help="", default=1000)
parser.add_argument('--n_topics',help="", default=2)
args = parser.parse_args()
def print_top_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
    return tword
def main():

    f = open(args.train_one_hot_vector_path)
    lines=f.readlines()
    list=[]
    for line in lines:
        list.append(line.strip())

    tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                    max_features=args.n_features,
                                    stop_words='english',
                                    max_df=0.95,
                                    min_df=1,                            
                                   )
    tf = tf_vectorizer.fit_transform(list)

    print('lda now topic:',args.n_topics)
    lda = LatentDirichletAllocation(n_components=args.n_topics, max_iter=300,
                                    learning_method='batch',
                                    learning_offset=100,
                                    # doc_topic_prior=0.2,#1/K
                                    # topic_word_prior=0.2,#1/K
                                    random_state=666)
    lda.fit(tf)

    n_top_words = 40
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    topic_word = print_top_words(lda, tf_feature_names, n_top_words)
    topics = lda.transform(tf)
    print(topics[0])
    s = str(topics[0]) + '\n'
    f = open(args.output_path, 'w')
    for i in range(len(topics)):
        s=str(topics[i]) + '\n'
        s=s[1:-2]
        ss=''
        for j in range(len(s)):
            if s[j]==' ' and s[j-1]==' ':
                continue
            if s[j]=='\n':
                continue
            ss=ss+s[j]
        ss=ss+'\n'
        if i==0 or i==len(topics)-1:
            print(ss)
        f.write(ss)
    f.close()
    print(len(lines))





if __name__ == '__main__':
    main()
