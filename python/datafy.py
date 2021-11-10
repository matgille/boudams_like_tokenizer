import random
import unicodedata

import torch
import utils as utils


class Datafier:
    def __init__(self, timestamp):
        self.max_length_examples = 0
        self.frequency_dict = {}
        self.unknown_threshold = 14  # Under this frequency the tokens will be tagged as <UNK>
        self.input_vocabulary = {}
        self.target_vocabulary = {}
        self.train_padded_examples = []
        self.train_padded_targets = []
        self.test_padded_examples = []
        self.test_padded_targets = []
        self.timestamp = timestamp

    def create_train_corpus(self, train_path):
        print("Create train corpus")
        imported_data = self.get_data_from_txt(train_path)
        self.input_vocabulary, self.target_vocabulary = self.create_vocab(imported_data)
        augmented_data = self.augment_data(imported_data, double_corpus=True)
        train_examples, train_targets = self.produce_train_corpus(augmented_data)
        self.train_padded_examples, self.train_padded_targets = self.pad_and_numerize(train_examples, train_targets)

    def create_test_corpus(self, test_path):
        """
        This function creates the test corpus, and uses the vocabulary of the train set to do so.
        Outputs: tensorized input, tensorized target, formatted input to ease accuracy computation.
        """
        print("Create test corpus")
        inputs = self.get_data_from_txt(test_path)
        treated_inputs = self.augment_data(inputs, double_corpus=False)
        test_examples, test_targets = self.produce_train_corpus(treated_inputs)
        self.test_padded_examples, self.test_padded_targets = self.pad_and_numerize(test_examples, test_targets)

    def get_frequency(self, data_as_string):
        for char in data_as_string:
            try:
                self.frequency_dict[char] += 1
            except:
                self.frequency_dict[char] = 1

    def get_data_from_txt(self, path: str) -> list:
        '''
        Import data and normalize
        '''
        with open(path, "r") as training_file:
            imported_data = training_file.read()
            cleaned_text = [utils.remove_multiple_spaces(line) for line in imported_data.split("\n")]
            normalized = [utils.normalize(line) for line in cleaned_text]
        self.get_frequency("".join(normalized))
        return normalized

    def augment_data(self, data: list, double_corpus=True) -> list:
        '''
        This function takes the data set and randomly modifies its segmentation to produce the targets
        :param double_corpus: If set to True, the data will be doubled, and then augmented
        '''
        if double_corpus:
            data = data + data
        random.shuffle(data)
        augmented_data = []
        spaces_list = ["<s-s>", "<s-S>", "<S-s>"]
        for example in data:
            intermed_list = ["<SOS>"]
            for idx, char in enumerate(example):
                if char == " ":
                    true_or_false = utils.random_bool(50)
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
                    if utils.random_bool(10) and example[idx - 1] not in spaces_list:
                        intermed_list.append("<s-S>")

            intermed_list.append("<EOS>")
            augmented_data.append(intermed_list)
        return augmented_data

    def produce_train_corpus(self, augmented_data: list) -> tuple:
        """
        This function takes the targets and creates the examples
        """
        examples = []
        targets = []
        for element in augmented_data:
            example = []
            target = []
            n = 0
            for idx, char in enumerate(element):
                if char == "<S-s>":
                    target[n - 1] = "<S-s>"
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
        if max_length_targets > 300:
            print("There is a problem with some line way too long. Please check the datasets.")
            exit(0)
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
                             "<S-s>": 4, # No space > space (equivalent to <WB> in boudams)
                             "<s-s>": 5, # space > space
                             "<s-S>": 6} # space > no space

        n = 5
        data_string = "".join(data).replace(" ", "")
        for char in data_string:
            try:
                input_vocabulary[char] == n
            except:
                input_vocabulary[char] = n
                n += 1
        torch.save(input_vocabulary, f"../models/input_vocab_{self.timestamp}.voc")
        print(input_vocabulary)
        return input_vocabulary, target_vocabulary
