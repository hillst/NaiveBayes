#!/bin/bash
data_dir=data
python NaiveBayes.py -v $data_dir/vocabulary.txt -n $data_dir/newsgrouplabels.txt -T $data_dir/train.data -t $data_dir/test.data -L $data_dir/train.label -l $data_dir/test.label
