import sys
import random
import torch
import numpy as np
import python.utils as utils


class Datafier:
    def __init__(self):
        print("Preparing corpus.")
        self.max_length_examples = 0
        self.frequency_dict = {}
        self.unknown_threshold = 14  # Under this frequency the tokens will be tagged as <UNK>

    def create_corpus(self, path):
        imported_data = self.get_data_from_txt(path)
        self.input_vocabulary, self.target_vocabulary = self.create_vocab(imported_data)
        self.reverse_input_vocab = {v: k for k, v in self.input_vocabulary.items()}
        self.reverse_target_vocab = {v: k for k, v in self.target_vocabulary.items()}
        augmented_data = self.augment_data(imported_data)
        examples, targets = self.produce_train_corpus(augmented_data)
        self.padded_examples, self.padded_targets = self.pad_and_numerize(examples, targets)

    def get_frequency(self, data_as_string):
        for char in data_as_string:
            try:
                self.frequency_dict[char] += 1
            except:
                self.frequency_dict[char] = 1
        print(self.frequency_dict)

    def get_data_from_txt(self, path):
        with open(path, "r") as training_file:
            imported_data = training_file.read()
        self.get_frequency(imported_data.replace("\n", ""))
        imported_data = imported_data.split("\n")
        return imported_data

    def random_bool(self, probs: int):
        """
        This function returns True with a probability given
        by the param probs
        """
        number = random.randint(0, 100)
        if number < probs:
            result = True
        else:
            result = False
        return result

    def augment_data(self, data):
        data = data + data
        random.shuffle(data)
        augmented_data = []
        spaces_list = ["<s-s>", "<s-S>", "<S-s>"]
        for example in data:
            intermed_list = ["<SOS>"]
            for idx, char in enumerate(example):
                if char == " ":
                    true_or_false = self.random_bool(50)
                    if true_or_false:
                        intermed_list.append("<S-s>")
                    elif not true_or_false and example[idx - 1] not in spaces_list:
                        intermed_list.append("<s-s>")
                else:
                    if self.frequency_dict[char] < self.unknown_threshold:
                        intermed_list.append("<UNK>")
                    else:
                        intermed_list.append(char)
                    # We add a space that should not exist
                    if self.random_bool(10) and example[idx - 1] not in spaces_list:
                        intermed_list.append("<s-S>")

            intermed_list.append("<EOS>")
            augmented_data.append(intermed_list)
        return augmented_data


    def produce_train_corpus(self, augmented_data):
        examples = []
        targets = []
        for element in augmented_data:
            example = []
            target = []
            n = 0
            for idx, char in enumerate(element):
                if char == "<S-s>":
                    target[n - 1] = "<WB>"
                elif char in ['<SOS>', '<EOS>']:
                    example.append(char)
                    target.append(char)
                    n += 1
                elif char not in ['<s-S>', '<s-s>']:
                    example.append(char)
                    target.append("<WC>")
                    n += 1
                elif char in ['<s-S>', '<s-s>']:
                    example.append("<S>")
                    target.append(char)
                    n += 1
            examples.append(example)
            targets.append(target)

        return examples, targets

    def pad_and_numerize(self, examples, targets):
        self.max_length_examples = max([len(example) for example in examples])
        max_length_targets = max([len(target) for target in targets])
        pad_value = "<PAD>"
        padded_examples = []
        padded_targets = []
        for example in examples:
            example_length = len(example)
            example = example + [pad_value for _ in range(self.max_length_examples - example_length)]
            example = ["<PAD>"] + example
            example = [self.input_vocabulary[char] for char in example]
            padded_examples.append(example)

        for target in targets:
            target_length = len(target)
            target = target + [pad_value for _ in range(max_length_targets - target_length)]
            target = ["<PAD>"] + target
            target = [self.target_vocabulary[char] for char in target]
            padded_targets.append(target)
        return padded_examples, padded_targets

    def create_vocab(self, data):
        input_vocabulary = {"<PAD>": 0,
                            "<SOS>": 1,
                            "<EOS>": 2,
                            "<S>": 3,
                            "<UNK>": 4}

        target_vocabulary = {"<PAD>": 0,
                             "<SOS>": 1,
                             "<EOS>": 2,
                             "<WC>": 3,
                             "<WB>": 4,
                             "<s-s>": 5,
                             "<s-S>": 6}

        n = 5
        data_string = "".join(data).replace(" ", "")
        for char in data_string:
            try:
                input_vocabulary[char] == n
            except:
                input_vocabulary[char] = n
                n += 1
        torch.save(input_vocabulary, "models/input_vocab.voc")
        torch.save(target_vocabulary, "models/target_vocab.voc")
        return input_vocabulary, target_vocabulary

    def compute_accuracy(self, inputs, gt, preds, batch_size):
        # preds shape: [batch_size*max_length, output_dim]
        highger_prob = torch.topk(preds, 1).indices
        # shape [batch_size*max_length, 1]: list of all characters in batch

        # Shape [batch_size*max_length]
        list_of_predictions = highger_prob.view(-1).tolist()

        mask = [self.reverse_target_vocab[pred] for pred in list_of_predictions]
        splitted_mask = np.split(np.array(mask), batch_size)

        # Length: batch_size lists of max_length size.
        splitted_mask = [sentence.tolist() for sentence in splitted_mask]
        batch_accuracy = 0
        for sentence_number in range(batch_size):
            current_mask = splitted_mask[sentence_number]
            # Length: max_length
            current_input = [self.reverse_input_vocab[char] for char in inputs[sentence_number]]
            # Length: max_length
            current_target = [self.reverse_target_vocab[trg] for trg in gt[sentence_number]]

            # We split the mask in n examples.
            padding_position = utils.find(current_input, "<PAD>")
            eos_position = utils.find(current_input, "<EOS>")
            try:
                min_pos = padding_position[0]
                max_pos = eos_position[0]
            except:
                print("Input error. ")
                exit(0)
            min_pos += 2
            max_pos -= 1
            sentence = current_input[min_pos: max_pos]
            prediction = current_mask[min_pos: max_pos]
            target = current_target[min_pos: max_pos]
            sentence_masked_zipped = list(zip(sentence, prediction))
            prediction_target_zipped = list(zip(prediction, target))
            good_predictions = 0
            number_of_items = len(prediction_target_zipped)
            for prediction, target in prediction_target_zipped:
                if prediction == target:
                    good_predictions += 1
                else:
                    continue
            sentence_accuracy = good_predictions / number_of_items
            batch_accuracy += sentence_accuracy

            final_pred = []
            for char, mask in sentence_masked_zipped:
                if mask == "<WC>":
                    final_pred.append(char)
                elif mask == "<s-s>":
                    final_pred.append(" ")
                elif mask == "<WB>":
                    final_pred.append(char)
                    final_pred.append("&rien-esp;")
                elif mask == "<s-S>":
                    final_pred.append("&esp-rien;")

            orig_sentence = "".join([char if char != "<S>" else " " for char in sentence])
            final_pred = "".join(final_pred)
            # print(f"Sentence to process: {orig_sentence}")
            # print(f"Prediction: {final_pred}")
        final_accuracy = batch_accuracy / batch_size
        return final_accuracy
        # print(f"Batch accuracy: {final_accuracy}")
