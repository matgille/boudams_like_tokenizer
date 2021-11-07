from datetime import datetime
import glob
import os
import random
import shutil
import numpy as np
import torch
import utils as utils
import seq2seq as seq2seq
import model as modele
import datafy as datafy
import tqdm
from statistics import mean


class Trainer:
    def __init__(self, epochs, lr, batch_size, train_path, test_path):
        # First we prepare the corpus
        print("Preparing corpus.")
        now = datetime.now()
        self.timestamp = now.strftime("%d-%m-%Y_%H:%M:%S")
        datafyer = datafy.Datafier(self.timestamp)
        datafyer.create_train_corpus(train_path)
        self.train_examples = datafyer.train_padded_examples
        self.train_targets = datafyer.train_padded_targets
        # Il manque à importer le set de test.
        datafyer.create_test_corpus(test_path)
        self.test_examples = datafyer.test_padded_examples
        self.test_targets = datafyer.test_padded_targets

        print(f"Number of train examples: {len(self.train_examples)}")
        print(f"Number of test examples: {len(self.test_examples)}")

        self.input_vocab = datafyer.input_vocabulary
        self.target_vocab = datafyer.target_vocabulary
        self.reverse_input_vocab = {v: k for k, v in self.input_vocab.items()}
        self.reverse_target_vocab = {v: k for k, v in self.target_vocab.items()}

        self.corpus_size = len(self.train_examples)
        self.steps = self.corpus_size // batch_size
        INPUT_DIM = len(self.input_vocab)
        OUTPUT_DIM = len(self.target_vocab)
        self.device = 'cuda:0'
        # self.device = 'cpu'
        EMB_DIM = 256
        HID_DIM = 256  # each conv. layer has 2 * hid_dim filters
        ENC_LAYERS = 10  # number of conv. blocks in encoder
        ENC_KERNEL_SIZE = 5  # must be odd!
        ENC_DROPOUT = 0.25
        self.TRG_PAD_IDX = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.enc = modele.Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, self.device)
        self.dec = modele.LinearDecoder(EMB_DIM, OUTPUT_DIM)
        self.model = seq2seq.Seq2Seq(self.enc, self.dec).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.TRG_PAD_IDX)

        self.accuracies = []

    def save_model(self, epoch):
        torch.save(self.model, f"../models/.tmp/model_tokenizer_{epoch}.pt")

    def get_best_model(self):
        print(self.accuracies)
        best_epoch_accuracy = self.accuracies.index(max(self.accuracies))
        print(f"Best model: {best_epoch_accuracy}.")
        models = glob.glob(f"../models/.tmp/model_tokenizer_*.pt")
        for model in models:
            if model == f"../models/.tmp/model_tokenizer_{best_epoch_accuracy}.pt":
                shutil.move(model, f"../models/model_tokenizer.best_{best_epoch_accuracy}_{self.timestamp}.pt")
            else:
                os.remove(model)
        print(f"Saving best model to ../models/model_tokenizer.best_{best_epoch_accuracy}_{self.timestamp}.pt")

    def train(self, clip=0.1, shuffle_dataset=True):
        print("Starting training")
        self.model.train()
        epoch_loss = 0
        if shuffle_dataset:
            example_target = list(zip(self.train_examples, self.train_targets))
            random.shuffle(example_target)
            self.train_examples, self.train_targets = zip(*example_target)
        print("Evaluating randomly intiated model")
        self.evaluate()
        for epoch in range(self.epochs):
            epoch_number = epoch + 1
            print(f"Epoch {str(epoch_number)}")
            epoch_accuracies = []
            for i in tqdm.tqdm(range(self.steps), unit_scale=self.batch_size):
                examples = self.train_examples[self.batch_size * i:self.batch_size * (i + 1)]
                targets = self.train_targets[self.batch_size * i:self.batch_size * (i + 1)]
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

                # output = [batch size * trg len - 1, output dim]
                # trg = [batch size * trg len - 1]
                loss = self.criterion(output, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimizer.step()
                epoch_loss += loss.item()

            # We compute accuracy on last batch of data
            self.evaluate()
            print(f"Loss: {str(loss.item())}")
            self.save_model(epoch_number)
        self.get_best_model()


    def evaluate(self):
        """
        Réécrire la fonction pour comparer directement target et prédiction pour
        produire l'accuracy.
        """
        value = random.randint(0, len(self.test_examples) - 128)
        examples = self.test_examples[value:value + 128]
        targets = self.test_targets[value:value + 128]

        tensor_examples = utils.tensorize(examples).to(self.device)
        tensor_target = utils.tensorize(targets).to(self.device)
        with torch.no_grad():
            preds = self.model(tensor_examples)
            # Loss calculation
            output_dim = preds.shape[-1]
            output = preds.contiguous().view(-1, output_dim)
            trg = tensor_target.contiguous().view(-1)
            loss = self.criterion(output, trg)

        highger_prob = torch.topk(preds, 1).indices
        # shape [batch_size*max_length, 1]: list of all characters in batch
        correct_predictions = 0
        examples_number = 0
        for i, target in enumerate(targets):
            mask = [element[0] for element in highger_prob.tolist()[i]]
            zipped = list(zip(mask, target))

            # We have to exclude the evaluation when target is <PAD> because the network has ignored when training;
            # We ignore them too.
            for prediction, target_mask in zipped:
                examples_number += 1
                if target_mask == self.TRG_PAD_IDX:
                    examples_number -= 1
                if prediction == self.TRG_PAD_IDX:
                    pass
                elif prediction == target_mask:
                    correct_predictions += 1

        accuracy = correct_predictions / examples_number
        self.accuracies.append(accuracy)
        print(f"Loss: {loss}\nAccuracy on test set: {accuracy}")
