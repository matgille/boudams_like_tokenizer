import glob
import os
import random
import argparse
import json

import torch

import trainer as trainer
import tagger as tagger

# https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb




seed = 1234
random.seed(seed)

entities_mapping = {"add_space": "&rien-esp;",
                    "remove_space": "&esp-rien;"}

parser = argparse.ArgumentParser()
# https://stackoverflow.com/a/30493366 Optional positional arguments
parser.add_argument("mode", help="Tokenizer mode: train, tag xml, if"
                                 ".")
parser.add_argument("-ft", "--fine_tune", default=False,
                    help="Fine-tuning mode: use existing vocab and model.")
parser.add_argument("-f", "--files", default=False,
                    help="File to tag.", nargs='+')
parser.add_argument('-o', "--output", help="Output folder")
parser.add_argument('-d', '--device', help="Device to be used", default='cuda:0')
parser.add_argument('-e', '--entities', help="Produce XML entities", default=False)
parser.add_argument('-ep', '--epochs', help="Number of epochs", default=10, type=int)
parser.add_argument('-k', '--kernel_size', help="Kernel size", default=5, type=int)
parser.add_argument('-b', '--batch_size', help="Sets batch size", default=128, type=int)
parser.add_argument('-p', '--parameters', help="Path to params files")
parser.add_argument('-v', '--vocabulary', help="Path to vocabulary", default=None)
parser.add_argument('-m', '--model', help="Path to model", default=None)
parser.add_argument('-rb', '--remove_breaks', help="Remove line breaks", default=False)
parser.add_argument('-w', '--workers', help="Number of workers for dataloading", default=0, type=int)
parser.add_argument('-ho', '--hyphens_only', help="Detect only hyphens when tagging", default=False)


args = parser.parse_args()
fine_tune = True if args.fine_tune == "True" else False
batch_size = args.batch_size
entities = args.entities
mode = args.mode
files = args.files
output_dir = args.output
remove_breaks = args.remove_breaks
parameters = args.parameters
device = args.device
lb_only = (args.hyphens_only == "True")
workers = int(args.workers)
epochs = args.epochs
kernel_size = args.kernel_size

try:
    os.mkdir(output_dir)
except:
    pass

if mode == 'train':
    with open(parameters, "r") as conf_file:
        conf_dict = json.load(conf_file)
    train_path = conf_dict["train_path"]
    test_path = conf_dict["test_path"]
    model = conf_dict["model"]
    vocab = conf_dict["vocab"]


    print("Starting training")
    trainer = trainer.Trainer(batch_size=batch_size,
                              epochs=epochs,
                              lr=0.0005,
                              device=device,
                              train_path=train_path,
                              test_path=test_path,
                              fine_tune=fine_tune,
                              model=model,
                              vocab=vocab,
                              output_dir=output_dir, 
                              workers=workers, 
                              kernel_size=kernel_size)
    trainer.train()


elif mode == 'tag_xml':
    vocab = args.vocabulary
    model = args.model
    if not files:
        print(f"Please indicate an input file.")
        exit(0)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tagger = tagger.Tagger(device=device,
                           input_vocab=vocab,
                           model=model,
                           remove_breaks=False,
                           xml_entities=entities,
                           entities_mapping=entities_mapping,
                           debug=False,
                           lb_only=lb_only)
    for file in files:
        tagger.tokenize_xml(file, batch_size)



elif mode == 'tag_txt':
    vocab = args.vocabulary
    model = args.model
    if not files:
        print(f"Please indicate an input file.")
        exit(0)
    tagger = tagger.Tagger(device=device,
                           input_vocab=vocab,
                           model=model,
                           remove_breaks=remove_breaks,
                           debug=False,
                           lb_only=lb_only)
    for file in files:
        tagger.tokenize_txt(file)
