import os
import random
import sys
import argparse

import trainer as trainer
import tagger as tagger

# https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb

train_path = "/home/mgl/Bureau/These/datasets/segmentation_segmentor/datasets/regimiento_gold/train/train.txt"
test_path = "/home/mgl/Bureau/These/datasets/segmentation_segmentor/datasets/regimiento_gold/test/test.txt"

train_path = "/home/mgl/Bureau/These/datasets/segmentation_segmentor/datasets/a_z_s_gold/train/train.txt"
test_path = "/home/mgl/Bureau/These/datasets/segmentation_segmentor/datasets/a_z_s_gold/test/test.txt"

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
parser.add_argument('-o', "--output", help="Output folder")
parser.add_argument('-d', '--device', help="Device to be used", default='cuda:0')
parser.add_argument('-e', '--entities', help="Produce XML entities", default=False)
parser.add_argument('-b', '--batch_size', help="Sets batch size", default=64, type=int)

output_dir = "../models/test_a_z"

args = parser.parse_args()
fine_tune = args.fine_tune
batch_size = args.batch_size
mode = args.mode
file = args.file
output_dir = args.output
device = args.device

try:
    os.mkdir(output_dir)
except:
    pass

vocab = "../models/saved/a_z/vocab.voc"
model = "../models/saved/a_z/best.pt"

vocab = "../models/saved/val_s/vocab.voc"
model = "../models/saved/val_s/best.pt"

vocab = "../models/saved/a_z_s/vocab.voc"
model = "../models/.tmp/model_tokenizer_2.pt"

if mode == 'train':
    print("Starting training")
    trainer = trainer.Trainer(batch_size=batch_size,
                              epochs=15,
                              lr=0.0005,
                              device=device,
                              train_path=train_path,
                              test_path=test_path,
                              fine_tune=fine_tune,
                              model=model,
                              vocab=vocab,
                              output_dir=output_dir)
    trainer.train(shuffle_dataset=True)
    print(trainer.input_vocab)


elif mode == 'tag_xml':
    if not file:
        print(f"Please indicate an input file.")
        exit(0)
    tagger = tagger.Tagger(device=device,
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
    tagger = tagger.Tagger(device=device,
                           input_vocab=vocab,
                           model=model,
                           remove_breaks=False,
                           debug=False)
    tagger.tokenize_txt(file)
