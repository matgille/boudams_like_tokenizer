import random

import numpy as np
import torch
import python.utils as utils
import python.seq2seq as seq2seq
import python.model as modele
import python.datafy as datafy
import tqdm
from statistics import mean

class Trainer:
    def __init__(self, epochs, lr, batch_size, corpus):
        # First we prepare the corpus
        self.datafyer = datafy.Datafier()
        self.datafyer.create_corpus(corpus)
        self.examples = self.datafyer.padded_examples
        self.targets = self.datafyer.padded_targets
        self.input_vocab = self.datafyer.input_vocabulary
        self.target_vocab = self.datafyer.target_vocabulary
        assert len(self.examples) == len(self.targets), 'Examples and target must have the same length'
        self.corpus_size = len(self.examples)
        self.iterator = self.corpus_size // batch_size
        INPUT_DIM = len(self.input_vocab)
        OUTPUT_DIM = len(self.target_vocab)
        self.device = 'cuda:0'
        EMB_DIM = 256
        HID_DIM = 256  # each conv. layer has 2 * hid_dim filters
        ENC_LAYERS = 5  # number of conv. blocks in encoder
        ENC_KERNEL_SIZE = 3 # must be odd!
        ENC_DROPOUT = 0.25
        TRG_PAD_IDX = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.enc = modele.Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, self.device)
        self.dec = modele.LinearDecoder(EMB_DIM, OUTPUT_DIM)
        self.model = seq2seq.Seq2Seq(self.enc, self.dec).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


    def eval(self):
        with torch.no_grad:
            pass


    def save_model(self):
        torch.save(self.model, "models/model_tokenizer.pt")


    def train(self, clip=0.1, shuffle_dataset=True):
        self.model.train()
        epoch_loss = 0
        if shuffle_dataset:
            example_target = list(zip(self.examples, self.targets))
            random.shuffle(example_target)
            self.examples, self.targets = zip(*example_target)
        for epoch in range(self.epochs):
            epoch_number = epoch + 1
            print(f"Epoch {str(epoch_number)}")
            epoch_accuracies = []
            for i in tqdm.tqdm(range(self.iterator), unit_scale=self.batch_size):
                examples = self.examples[self.batch_size*i:self.batch_size*(i+1)]
                targets = self.targets[self.batch_size*i:self.batch_size*(i+1)]
                tensor_examples = utils.tensorize(examples)
                tensor_targets = utils.tensorize(targets)


                # Shape [batch_size, max_length]
                tensor_examples = tensor_examples.to(self.device)
                # Shape [batch_size, max_length]
                tensor_targets = tensor_targets.to(self.device)

                self.optimizer.zero_grad()

                # Shape [batch_size, max_length, output_dim]
                output = self.model(tensor_examples)
                output_dim = output.shape[-1]

                # Shape [batch_size*max_length, output_dim]
                output = output.contiguous().view(-1, output_dim)

                # Shape [batch_size*max_length]
                trg = tensor_targets.contiguous().view(-1)

                batch_accuracy = self.datafyer.compute_accuracy(examples, targets, output, self.batch_size)
                epoch_accuracies.append(batch_accuracy)
                # output = [batch size * trg len - 1, output dim]
                # trg = [batch size * trg len - 1]
                loss = self.criterion(output, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f"Loss: {str(loss)}")
            print(f"Mean accuracy for epoch {str(epoch_number)}: {str(mean(epoch_accuracies))}")
        self.save_model()


