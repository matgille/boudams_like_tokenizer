import random
from torch.utils.data import Dataset
import torch
import utils as utils


# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CustomTextDataset(Dataset):
    def __init__(self, mode, train_path, test_path, fine_tune, input_vocab, max_length, device, all_dataset_on_device):
        self.datafy = Datafier(train_path, test_path, fine_tune, input_vocab, max_length)
        self.mode = mode
        if mode == "train":
            self.datafy.create_train_corpus()
        else:
            self.datafy.create_test_corpus()
        if all_dataset_on_device:
            self.datafy.train_padded_examples = torch.LongTensor(self.datafy.train_padded_examples).to(device)
            self.datafy.test_padded_examples = torch.LongTensor(self.datafy.test_padded_examples).to(device)
            self.datafy.train_padded_targets = torch.LongTensor(self.datafy.train_padded_targets).to(device)
            self.datafy.test_padded_targets = torch.LongTensor(self.datafy.test_padded_targets).to(device)

    def __len__(self):
        if self.mode == "train":
            return len(self.datafy.train_padded_examples)
        else:
            return len(self.datafy.test_padded_examples)

    def __getitem__(self, idx):
        if self.mode == "train":
            examples = self.datafy.train_padded_examples[idx]
            labels = self.datafy.train_padded_targets[idx]
        else:
            examples = self.datafy.test_padded_examples[idx]
            labels = self.datafy.test_padded_targets[idx]
        return examples, labels


class Datafier:
    def __init__(self, train_path, test_path, fine_tune, input_vocab, max_length):
        self.max_length_examples = 0
        self.frequency_dict = {}
        self.unknown_threshold = 14  # Under this frequency the tokens will be tagged as <UNK>
        self.length_threshold = max_length
        self.input_vocabulary = {}
        self.target_vocabulary = {}
        self.train_padded_examples = []
        self.train_padded_targets = []
        self.test_padded_examples = []
        self.test_padded_targets = []
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = self.get_data_from_txt(train_path)
        self.test_data = self.get_data_from_txt(test_path)
        self.previous_model_vocab = input_vocab
        self.target_vocabulary = {"<PAD>": 0,
                                  "<SOS>": 1,
                                  "<EOS>": 2,
                                  "<WC>": 3,
                                  "<S-s>": 4,  # No space > space (equivalent to <WB> in boudams)
                                  "<s-s>": 5,  # space > space
                                  "<s-S>": 6}  # space > no space
        if fine_tune:
            self.input_vocabulary = input_vocab
            self.update_vocab(input_vocab)
        else:
            self.create_vocab(self.train_data, self.test_data)

    def update_vocab(self, input_vocab):
        """
        This function updates the existing vocab with the new examples.
        """
        length_previous_vocab: int = len(input_vocab)
        orig_list_of_characters: set = {char for char, _ in input_vocab.items()}
        train_data_as_string: str = self.get_txt_as_str(self.train_path)
        new_set_of_characters: set = utils.get_vocab(train_data_as_string)

        # We want the new characters, no matter if some chars are not present in the new vocab.
        unseen_chars = new_set_of_characters - orig_list_of_characters

        # We compare the original vocab with the new one to create a merged vocab
        # but we need to keep the order, i.e. to append new chars at the end of the dict
        if len(unseen_chars) != 0:
            for index, new_char in enumerate(list(unseen_chars)):
                self.input_vocabulary[new_char] = (length_previous_vocab + index)

    def create_vocab(self, train_data, test_data):
        input_vocabulary = {"<PAD>": 0,
                            "<SOS>": 1,
                            "<EOS>": 2,
                            "<S>": 3,
                            "<UNK>": 4}
        n = 5
        full_corpus = train_data + test_data
        data_string = "".join(full_corpus).replace(" ", "")
        for char in data_string:
            try:
                input_vocabulary[char] == n
            except:
                input_vocabulary[char] = n
                n += 1
        self.input_vocabulary = input_vocabulary

    def create_train_corpus(self):
        augmented_data = self.augment_data(self.train_data, double_corpus=False)
        train_examples, train_targets = self.produce_corpus(augmented_data)
        train_examples, train_targets = train_examples, train_targets
        train_padded_examples, train_padded_targets = self.pad_and_numerize(train_examples, train_targets)
        self.train_padded_examples = utils.tensorize(train_padded_examples)
        self.train_padded_targets = utils.tensorize(train_padded_targets)

    def create_test_corpus(self):
        """
        This function creates the test corpus, and uses the vocabulary of the train set to do so.
        Outputs: tensorized input, tensorized target, formatted input to ease accuracy computation.
        """
        treated_inputs = self.augment_data(self.test_data, double_corpus=False)
        test_examples, test_targets = self.produce_corpus(treated_inputs)
        if len(test_examples) > 3000:
            test_examples, test_targets = test_examples[:10_000], test_targets[:10_000]
        test_padded_examples, test_padded_targets = self.pad_and_numerize(test_examples, test_targets)
        self.test_padded_examples = utils.tensorize(test_padded_examples)
        self.test_padded_targets = utils.tensorize(test_padded_targets)

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
        normalized_length_list = []
        for example in normalized:
            if len(example) > self.length_threshold:
                normalized_length_list.append(example[self.length_threshold])
                normalized_length_list.append(example[self.length_threshold])
            else:
                normalized_length_list.append(example)
        return normalized_length_list

    def get_txt_as_str(self, path) -> str:
        with open(path, "r") as training_file:
            imported_data = training_file.read()
            cleaned_text = [utils.remove_multiple_spaces(line) for line in imported_data.split("\n")]
            normalized = [utils.normalize(line) for line in cleaned_text]

        return "".join(normalized)

    def augment_data(self, data: list, double_corpus=False) -> list:
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

    def produce_corpus(self, augmented_data: list) -> tuple:
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
