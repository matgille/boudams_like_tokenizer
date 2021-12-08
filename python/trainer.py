from datetime import datetime
import glob
import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import utils as utils
import seq2seq as seq2seq
import model as modele
import datafy as datafy
import tqdm
from statistics import mean
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, epochs, lr, device, batch_size, train_path, test_path, fine_tune, output_dir,
                 **pretrained_params):
        # First we prepare the corpus
        now = datetime.now()
        self.device = device

        self.timestamp = now.strftime("%d-%m-%Y_%H:%M:%S")
        train_dataloader = datafy.CustomTextDataset("train", train_path, test_path)
        test_dataloader = datafy.CustomTextDataset("test", train_path, test_path)
        self.loaded_train_data = DataLoader(train_dataloader,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True)
        self.loaded_test_data = DataLoader(test_dataloader,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=True)

        self.output_dir = output_dir

        print(f"Number of train examples: {len(train_dataloader.datafy.train_padded_examples)}")
        print(f"Number of test examples: {len(test_dataloader.datafy.test_padded_examples)}")

        self.input_vocab = train_dataloader.datafy.input_vocabulary
        self.reverse_input_vocab = {v: k for k, v in self.input_vocab.items()}
        self.target_vocab = train_dataloader.datafy.target_vocabulary
        self.reverse_target_vocab = {v: k for k, v in self.target_vocab.items()}


        if fine_tune:
            self.pretrained_model = pretrained_params.get('model', None)
            self.pretrained_vocab = pretrained_params.get('vocab', None)
            if self.device == 'cpu':
                self.pre_trained_model = torch.load(self.pretrained_model, map_location=self.device)
            else:
                self.pre_trained_model = torch.load(self.pretrained_model).to(self.device)
            self.pretrained_vocab = torch.load(self.pretrained_vocab)
            pre_trained_weights = self.pre_trained_model.encoder.tok_embedding.weight
            embs_dim = pre_trained_weights.shape[1]

            # We compare the original vocab with the new one to create a merged vocab
            # but we need to keep the order, i.e. to append new chars at the end of the dict
            different_chars_pretrained = list(self.pretrained_vocab.keys())
            different_chars_target = list(self.input_vocab.keys())
            length_pretrained = len(self.pretrained_vocab)
            dict_to_update = self.pretrained_vocab.copy()
            # We only need to expand the vocab: we don't care if the new vocab
            # is less rich than the pretrained one
            new_chars = set(different_chars_target) - set(different_chars_pretrained)
            number_new_chars = len(new_chars)
            print(number_new_chars)
            if len(new_chars) != 0:
                for index, new_char in enumerate(list(new_chars)):
                    dict_to_update[new_char] = (length_pretrained + index)
            # We re-write input vocab and reverse input vocab
            self.input_vocab = dict_to_update.copy()
            self.reverse_input_vocab = {v: k for k, v in self.input_vocab.items()}

            # We create the updated embs:
            # First we create randomly initiated tensors corresponding to the number of new chars in the new dataset
            new_vectors = torch.FloatTensor(number_new_chars, embs_dim).to(self.device)

            # We then add the new vectors to the pre trained weights
            pretrained_and_new_tensors = torch.cat((pre_trained_weights, new_vectors), 0)

        self.corpus_size = train_dataloader.__len__()
        self.steps = self.corpus_size // batch_size

        self.test_steps = test_dataloader.__len__() // batch_size
        INPUT_DIM = len(self.input_vocab)
        OUTPUT_DIM = len(self.target_vocab)
        EMB_DIM = 256
        HID_DIM = 256  # each conv. layer has 2 * hid_dim filters
        ENC_LAYERS = 10  # number of conv. blocks in encoder
        ENC_KERNEL_SIZE = 11  # must be odd!
        ENC_DROPOUT = 0.25
        self.TRG_PAD_IDX = self.target_vocab["<PAD>"]
        self.epochs = epochs
        self.batch_size = batch_size

        if fine_tune:
            # If fine tune is True, we take the pre-trained model and modify its embedding layer to match
            # new vocabulary
            self.model = self.pre_trained_model
            self.model.encoder.tok_embedding = nn.Embedding.from_pretrained(pretrained_and_new_tensors)
        else:
            # If not, just initialize a new seq2seq model.
            self.enc = modele.Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT,
                                      self.device)
            self.dec = modele.LinearDecoder(EMB_DIM, OUTPUT_DIM)
            self.model = seq2seq.Seq2Seq(self.enc, self.dec)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.TRG_PAD_IDX)
        print(self.model.__repr__())
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
                shutil.move(model, f"{self.output_dir}/best.pt")
            else:
                os.remove(model)
        print(f"Saving best model to {self.output_dir}/best.pt")

    def train(self, clip=0.1, shuffle_dataset=True):
        print("Starting training")
        torch.save(self.input_vocab, f"{self.output_dir}/vocab.voc")
        self.model.train()
        epoch_loss = 0
        print("Evaluating randomly intiated model")
        self.evaluate()
        for epoch in range(self.epochs):
            epoch_number = epoch + 1
            print(f"Epoch {str(epoch_number)}")
            epoch_accuracies = []
            for examples, targets in tqdm.tqdm(self.loaded_train_data, unit_scale=self.batch_size):
                # Shape [batch_size, max_length]
                tensor_examples = examples.to(self.device)
                # Shape [batch_size, max_length]
                tensor_targets = targets.to(self.device)

                for param in self.model.parameters():
                    param.grad = None

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
            self.save_model(epoch_number)
        self.get_best_model()

    def evaluate(self):
        """
        Réécrire la fonction pour comparer directement target et prédiction pour
        produire l'accuracy.
        """
        print("Evaluating model on test data")
        epoch_accuracy = []
        epoch_loss = []
        for examples, targets in tqdm.tqdm(self.loaded_test_data, unit_scale=self.batch_size):
            # https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/3
            tensor_examples = examples.to(self.device, non_blocking=True)
            tensor_target = targets.to(self.device, non_blocking=True)
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
                predicted_class = [element[0] for element in highger_prob.tolist()[i]]
                zipped = list(zip(predicted_class, target))

                # We have to exclude the evaluation when target is <PAD> because the network has ignored when training;
                # We ignore them too.
                for prediction, target_class in zipped:
                    examples_number += 1
                    if target_class == self.TRG_PAD_IDX:
                        examples_number -= 1
                    if prediction == self.TRG_PAD_IDX:
                        pass
                    elif prediction == target_class:
                        correct_predictions += 1

            accuracy = correct_predictions / examples_number
            epoch_accuracy.append(accuracy)
            epoch_loss.append(loss.item())

        global_accuracy = mean(epoch_accuracy)
        global_loss = mean(epoch_loss)
        self.accuracies.append(global_accuracy)
        print(f"Loss: {global_loss}\nAccuracy on test set: {global_accuracy}")
