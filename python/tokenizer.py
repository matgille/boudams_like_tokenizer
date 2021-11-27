import random
import sys
import argparse

import trainer as trainer
import tagger as tagger

# https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb

train_path = "../.data/fro/train.tsv"
test_path = "../.data/fro/test.tsv"

train_path = "/home/mgl/Bureau/These/datasets/segmentation_segmentor/datasets/train/train.tsv"
test_path = "/home/mgl/Bureau/These/datasets/segmentation_segmentor/datasets/test/test.tsv"


train_path = "/home/mgl/Bureau/These/datasets/segmentation_segmentor/datasets/Val_S/train.txt"
test_path = "/home/mgl/Bureau/These/datasets/segmentation_segmentor/datasets/Val_S/test.txt"


seed = 1234
random.seed(seed)

entities_mapping = {"add_space": "&rien-esp;",
                    "remove_space": "&esp-rien;"}

parser = argparse.ArgumentParser()
# https://stackoverflow.com/a/30493366 Optional positional arguments
parser.add_argument("mode", help="Tokenizer mode: train, tag xml, tag txt.")
parser.add_argument("-ft", "--fine_tune", default=False,
                    help="Fine-tuning mode: use existing vocab and model.")
parser.add_argument("-f", "--file", default=False,
                    help="File to tag.")

args = parser.parse_args()
fine_tune = args.fine_tune
mode = args.mode
file = args.file

model = "../models/model_tokenizer.best_11_07-11-2021_20:38:28.pt"
vocab='../models/input_vocab_07-11-2021_20:38:28.voc'
device = 'cuda:0'
if mode == 'train':
    print("Starting training")
    trainer = trainer.Trainer(batch_size=64,
                              epochs=15,
                              lr=0.0005,
                              device=device,
                              train_path=train_path,
                              test_path=test_path,
                              fine_tune=fine_tune,
                              model=model,
                              vocab=vocab)
    trainer.train()


elif mode == 'tag_xml':
    if not file:
        print(f"Please indicate an input file.")
        exit(0)
    tagger = tagger.Tagger(device="cpu",
                           input_vocab=vocab,
                           model=model,
                           remove_breaks=False,
                           xml_entities=True,
                           entities_mapping=entities_mapping,
                           debug=False)

    tagger.tokenize_xml(file)



elif mode == 'tag_txt':
    if not file:
        print(f"Please indicate an input file.")
        exit(0)
    tagger = tagger.Tagger(device="cpu",
                           input_vocab="../models/input_vocab_07-11-2021_20:38:28.voc",
                           model="../models/model_tokenizer.best_11_07-11-2021_20:38:28.pt",
                           remove_breaks=False,
                           debug=False)

    tagger.tokenize_txt(file)
