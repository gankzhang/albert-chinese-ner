import os
import shutil
import random
from albert_ner import NerProcessor
import tensorflow as tf
import argparse
'''
From the data dir generate new training data with supervised part and unsupervised part.
'''
# parser = argparse.ArgumentParser(description='Test for argparse')
# parser.add_argument('--perc', '-n', default = 3)
# args = parser.parse_args()
# percentage = args.perc
flags = tf.flags

FLAGS = flags.FLAGS

percentage = FLAGS.perc

ori_data_dir = 'data'
new_data_dir = 'train_'+str(percentage)
processor = NerProcessor()
label_list = processor.get_labels()
train_examples = processor.get_train_examples(ori_data_dir)
if os.path.exists(new_data_dir):
    for file in os.listdir(new_data_dir):
        os.remove(os.path.join(new_data_dir,file))
    os.removedirs(new_data_dir)
os.mkdir(new_data_dir)
shutil.copyfile(os.path.join(ori_data_dir,'dev.txt'), os.path.join(new_data_dir,'dev.txt'))
shutil.copyfile(os.path.join(ori_data_dir,'test.txt'), os.path.join(new_data_dir,'test.txt'))

random.shuffle(train_examples)
num_of_sup_data = int(len(train_examples)*percentage/100) + 1
with open(os.path.join(new_data_dir,'train.txt'),'w',encoding='utf-8') as f:
    for id in range(num_of_sup_data):
        texts = train_examples[id].text.split(' ')
        labels = train_examples[id].label.split(' ')
        for text_id,text in enumerate(texts):
            f.write(' '.join([text,labels[text_id]])+'\n')
        f.write('\n')
with open(os.path.join(new_data_dir,'unsuper_train.txt'),'w',encoding='utf-8') as f:
    for id in range(num_of_sup_data,len(train_examples)):
        texts = train_examples[id].text.split(' ')
        labels = train_examples[id].label.split(' ')
        for text_id,text in enumerate(texts):
            f.write(' '.join([text,labels[text_id]])+'\n')
        f.write('\n')