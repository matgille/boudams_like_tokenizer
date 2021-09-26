import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import python.model as model
import python.trainer as trainer
import python.seq2seq as seq2seq
import python.utils as utils
import python.datafy as datafy
import python.tagger as tagger



import spacy
import numpy as np

import random
import math
import time

SEED = 1234
# https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True





test_string ="dieronlebatalla,evencieronleeganarondelcuantoquisieron.Desi\n" \
             "delrregnadodelrreydonalfonsoquefueenlaera\n" \
             "ylliricoagrandpriesaEescojodesuhuestea\n" \
             "eraaquellojusticiacomodereyede\n" \
             "ensucabodiez\n" \
             "trayçionquelefezieronlosmorosdetoledoquandorresçibieron\n" \
             "edixoaellosassi,segundcuentaJosefo:\n" \
             "nuevoera.PlogoaMercurioconestapregunta,e\n" \
             "fazieelmuchamaldattanmaloetanfalsofue\n" \
             "deviera,enonlapuedaperdernindexar\n" \
             "edeAaron,edecomomurioMoisen"

if sys.argv[1] == "train":
    trainer = trainer.Trainer(batch_size=128,
                              epochs=10,
                              lr=0.0005,
                              corpus=".data/train.txt")
    trainer.train()

elif sys.argv[1] == "tag":
    tagger = tagger.Tagger(input_vocab="models/input_vocab.voc",
                           target_vocab="models/target_vocab.voc",
                           model="models/model_tokenizer.pt",
                           verbosity=False)
    path = sys.argv[2]
    text = utils.text_to_string(path)
    tokenized = tagger.predict(text)
    utils.string_to_text(tokenized, path)
