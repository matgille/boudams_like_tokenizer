import random
import re
import unicodedata

import torch


def tensorize(array):
    tensorized_array = torch.tensor(array)
    return tensorized_array


def find(liste, char):
    return [i for i, ltr in enumerate(liste) if ltr == char]


def text_to_string(path):
    with open(path, "r") as input_file:
        text_as_string = input_file.read()
    return text_as_string


def string_to_text(string, path):
    path = path.replace(".txt", ".tokenized.txt")
    with open(path, "w") as output_file:
        output_file.write(string)


def normalize(line: str):
    # NFD semble la meilleure normalisation pour l'instant.
    # Voir https://unicode.org/reports/tr15/
    return "".join([unicodedata.normalize('NFD', char) for char in line])


def remove_multiple_spaces(text: str):
    return re.sub("\s+", " ", text)


def clean_and_normalize_encoding(line: str):
    if re.match(f'^\s+$', line):
        norm_line = None
    else:
        norm_line = re.sub("\s+", " ", line)
        norm_line = normalize(norm_line)
        # norm_line = "".join([char for char in norm_line])

    return norm_line


def random_bool(probs: int) -> bool:
    """
    This function returns True with a probability given
    by the param probs
    """
    number = random.randint(0, 100)
    if number < probs:
        return True
    else:
        return False


def entities_decl():
    "<!DOCTYPE entities_decl [" \
    "<!ENTITY esp-rien '<choice xmlns='http://www.tei-c.org/ns/1.0'><orig> </orig><reg/></choice>'>" \
    "<!ENTITY rien-esp '<choice xmlns='http://www.tei-c.org/ns/1.0'><orig/><reg><space/></reg></choice>'>" \
    "]>"
