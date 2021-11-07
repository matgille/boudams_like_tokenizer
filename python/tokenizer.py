import random
import sys

import trainer as trainer
import utils as utils
import tagger as tagger

# https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb


train_path = "/home/mgl/Bureau/These/datasets/segmentation_segmentor/datasets/train/train.tsv"
test_path = "/home/mgl/Bureau/These/datasets/segmentation_segmentor/datasets/test/test.tsv"

# train_path = "../.data/train.txt"
# test_path = "../.data/test.txt"
seed = 1234
random.seed(seed)
entities_mapping = {"add_space": "&rien-esp;",
                    "remove_space": "&esp-rien;"}
if sys.argv[1] == "train":
    trainer = trainer.Trainer(batch_size=64,
                              epochs=2,
                              lr=0.0005,
                              train_path=train_path,
                              test_path=test_path)
    trainer.train()


elif sys.argv[1] == "tag_xml":
    tagger = tagger.Tagger(device="cpu",
                           input_vocab="../models/input_vocab.voc",
                           target_vocab="../models/target_vocab.voc",
                           model="../models/model_tokenizer.pt",
                           remove_breaks=False,
                           xml_entities=True,
                           entities_mapping=entities_mapping,
                           debug=False)

    tagger.tokenize_xml(sys.argv[2])



elif sys.argv[1] == "tag_txt":
    tagger = tagger.Tagger(device="cuda:0",
                           input_vocab="../models/input_vocab.voc",
                           target_vocab="../models/target_vocab.voc",
                           model="../models/model_tokenizer.pt",
                           remove_breaks=False,
                           xml_entities=False,
                           entities_mapping=entities_mapping,
                           debug=False)

    tagger.tokenize_txt(sys.argv[2])