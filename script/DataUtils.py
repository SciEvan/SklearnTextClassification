#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:WWF
# datetime:2019/5/22 11:34
"""

"""
import os
import jieba
import re


class DataDeal:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def clean_str(self, string):
        strings = re.sub(r'[^\u4e00-\u9fa5]', " ", string)
        # string = ' '.join(strings.split())
        return strings

    def stop_word_list(self):
        stopwords = [line.strip() for line in open(self.data_dir + 'stopwords.txt', encoding='utf-8').readlines()]
        return stopwords

    def seg_depart(self, sentence):
        sentence_depart = jieba.cut(sentence.strip())
        stopwords = self.stop_word_list()
        outstr = ''
        for word in sentence_depart:
            if word not in stopwords:
                outstr += word
                outstr += " "
        return self.clean_str(outstr)

    def read_data(self, file_name):
        data_list = []
        with open(file_name, encoding='utf-8') as fn:
            for line in fn.readlines():
                line = self.seg_depart(line)
                line = ','.join(line.split(' ')).rstrip(',')
                data_list.append(line)
        return data_list

    # 数据预处理
    def process_data(self):
        data_name_list = os.listdir(self.data_dir)
        for data_name in data_name_list:
            if data_name == 'neg.txt':
                neg_data_list = self.read_data(self.data_dir + 'neg.txt')
                neg_data_label_list = [line.rstrip('\n') + '\t' + '0' for line in neg_data_list]

            if data_name == 'pos.txt':
                pos_data_list = self.read_data(self.data_dir + 'pos.txt')
                pos_data_label_list = [line.rstrip('\n') + '\t' + '1' for line in pos_data_list]
        data_list = neg_data_label_list + pos_data_label_list
        return data_list

    def write_data(self, written_file_name):
        data_list = self.process_data()
        with open(self.data_dir + written_file_name, 'w', encoding='utf-8') as wfn:
            for line in data_list:
                wfn.write(line + '\n')
        return ''


if __name__ == '__main__':
    dd = DataDeal('../data/')
    dd.write_data('merge1.txt')