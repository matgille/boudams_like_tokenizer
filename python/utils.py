import torch


def tensorize(array):
    tensorized_array = torch.tensor(array)
    return tensorized_array



def find(liste, char):
    return [i for i, ltr in enumerate(liste) if ltr == char]


def text_to_string(path):
    with open(path, "r") as input_file:
        text_as_string  = input_file.read()
    return text_as_string

def string_to_text(string, path):
    path = path.replace(".txt", ".tokenized.txt")
    with open(path, "w") as output_file:
        output_file.write(string)