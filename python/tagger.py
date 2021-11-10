import re
import shutil
import unicodedata

import lxml.etree as etree

import torch
import numpy as np
import tqdm

import utils as utils


# TODO: add an option to output break separated value.


class Tagger:
    """
    Tagger that manages xml and txt files.
    """

    def __init__(self,
                 device,
                 input_vocab: str,
                 model: str,
                 remove_breaks: bool,
                 debug: bool,
                 entities_mapping: dict = {},
                 xml_entities: bool = False):
        self.device = device
        if self.device == 'cpu':
            self.model = torch.load(model, map_location=self.device)
        else:
            self.model = torch.load(model).to(self.device)
        self.input_vocab = torch.load(input_vocab)
        self.target_vocab = {"<PAD>": 0,
                             "<SOS>": 1,
                             "<EOS>": 2,
                             "<WC>": 3,
                             "<S-s>": 4,
                             "<s-s>": 5,
                             "<s-S>": 6}
        self.reverse_input_vocab = {v: k for k, v in self.input_vocab.items()}
        self.reverse_target_vocab = {v: k for k, v in self.target_vocab.items()}
        self.remove_breaks = remove_breaks  # Keep the linebreaks in prediction?
        self.entities = xml_entities  # Whether to display original segmentation information with xml entities
        self.debug = debug
        self.entities_dict = entities_mapping

    def tokenize_xml(self, xml_file):
        """
        This function tokenizes an xml file
        """
        tei_namespace = 'http://www.tei-c.org/ns/1.0'
        namespace_declaration = {'tei': tei_namespace}
        with open(xml_file, "r") as input_file:
            parser = etree.XMLParser(resolve_entities=False)
            f = etree.parse(input_file, parser=parser)
        line_breaks = f.xpath("//tei:lb[not(parent::tei:fw)]", namespaces=namespace_declaration)
        text_lines = [utils.clean_and_normalize_encoding(line.tail) for line in line_breaks]
        with open(xml_file.replace('.xml', '.txt'), "w") as output_txt_file:
            output_txt_file.write("\n".join(text_lines))

        # To avoid out of memory problem.
        if len(text_lines) > 500:
            predictions = []
            batch_size = 256
            steps = len(text_lines) // batch_size
            for n in tqdm.tqdm(range(steps)):
                batch = text_lines[n * batch_size: (n * batch_size) + batch_size]
                predicted_batch = self.tag_and_detect_lb(batch)
                predictions.extend(predicted_batch)

            # We predict the last lines of the text that don't make it to a full batch.
            predictions.extend(self.tag_and_detect_lb(text_lines[(n + 1) * batch_size:-1]))
        else:
            predictions = self.tag_and_detect_lb(text_lines)

        zipped = list(zip(line_breaks, predictions))

        for index, (xml_element, (text, lb)) in enumerate(zipped[:-1]):
            # tei:lb stands for line beggining: we have to get the next line
            correct_element, (correct_text, next_lb) = zipped[index + 1]
            if lb:
                correct_element.set("break", "y")
            else:
                correct_element.set("break", "n")
            correct_element.tail = correct_text

        # Management of last tei:lb
        last_element, (last_text_node, last_lb) = zipped[-1]
        last_element.tail = last_text_node

        shutil.copy("XML/entities.dtd", xml_file.replace(".xml", ".dtd"))
        with open(xml_file.replace(".xml", ".tokenized.xml"), "w") as output_file:
            final_string = etree.tostring(f, pretty_print=True).decode().replace("amp;", "")

            # Producing the DTD declaration
            if self.entities:
                final_string = final_string.replace("<TEI",
                                                    f"<?xml version='1.0' encoding='UTF-8'?>\n"
                                                    f"<!DOCTYPE TEI SYSTEM '{xml_file.split('/')[-1].replace('.xml', '')}.dtd'>\n"
                                                    f"<TEI")
                final_string.replace("rien-esp", self.entities_dict["add_space"][1:-1])
                final_string.replace("esp-rien", self.entities_dict["remove_space"][1:-1])
            output_file.write(final_string)

    def tokenize_txt(self, txt_file):
        """
        This function tokenizes a txt file
        """
        with open(txt_file, "r") as input_file:
            inputs = [line for line in input_file.read().split("\n")]
            text_lines = [utils.clean_and_normalize_encoding(line) for line in inputs]
            joined = "".join(line for line in text_lines)
            get_chars = list(set([char for char in joined]))
            print(get_chars)
            text_lines = [line for line in text_lines if line is not None]
        with open(txt_file.replace(".txt", ".norm.txt"), "w") as output_norm_file:
            output_norm_file.write("\n".join(text_lines))

        # To avoid out of memory problem.
        if len(text_lines) > 500:
            predictions = []
            batch_size = 256
            steps = len(text_lines) // 256
            for n in tqdm.tqdm(range(steps)):
                batch = text_lines[n * batch_size: (n * batch_size) + batch_size]
                predicted_batch = self.tag_and_detect_lb(batch)
                predictions.extend(predicted_batch)

            # We predict the last lines of the text that don't make it to a full batch.
            predictions.extend(self.tag_and_detect_lb(text_lines[(n + 1) * batch_size:-1]))
        else:
            predictions = self.tag_and_detect_lb(text_lines)
        predictions = [f'{text}-' if line_break is False else text for (text, line_break) in predictions]
        with open(txt_file.replace('.txt', '.tokenized.txt'), "w") as input_file:
            input_file.write("\n".join(predictions))
        print("Done!")

    def predict_lines(self, lines_to_predict: list):
        """
        Makes prediction
        :param lines_to_predict: the sentence to predict
        returns: a list of tuples ('predicted line', bool) the latter being the
        decision to break the line or not.
        """
        input_tensor, formatted_inputs = self.lines_to_tensor(lines_to_predict)
        input_size = len(formatted_inputs)
        input_tensor = input_tensor.to(self.device)
        prediction = self.model(input_tensor)
        higher_prob = torch.topk(prediction, 1).indices

        # Shape [batch_size*max_length]
        list_of_predictions = higher_prob.view(-1).tolist()

        mask = [self.reverse_target_vocab[pred] for pred in list_of_predictions]
        splitted_mask = np.split(np.array(mask), input_size)

        # Length: batch_size lists of max_length size.
        splitted_mask = [sentence.tolist() for sentence in splitted_mask]
        result = []
        for sentence_number in range(input_size):
            current_mask = splitted_mask[sentence_number]
            # Length: max_length
            current_input = formatted_inputs[sentence_number]

            # We split the mask in n examples.
            padding_position = utils.find(current_input, "<PAD>")
            eos_position = utils.find(current_input, "<EOS>")
            try:
                min_pos = padding_position[0]
                max_pos = eos_position[0]
            except:
                print("Input error. ")
                exit(0)
            min_pos += 2  # As we got <PAD> and then <SOS> starting the prediction
            sentence = current_input[min_pos: max_pos]
            predict_mask = current_mask[min_pos: max_pos]
            sentence_masked_zipped = list(zip(sentence, predict_mask))
            predicted_line = []
            for char, mask in sentence_masked_zipped:
                if mask == "<WC>":
                    if char != "<S>":
                        predicted_line.append(char)
                    else:
                        predicted_line.append(" ")
                elif mask == "<s-s>":
                    # Ici si la machine apprend mal et reconnaît un
                    # espace où il n'y en a pas, on peut avoir des problèmes
                    if char == "<S>":
                        predicted_line.append(" ")
                    else:
                        predicted_line.append(char)
                elif mask == "<S-s>":
                    if char != "<S>":
                        predicted_line.append(char)
                    else:
                        predicted_line.append(" ")
                    if self.entities:
                        predicted_line.append(self.entities_dict["add_space"])
                    else:
                        predicted_line.append(" ")
                elif mask == "<s-S>":
                    # Ici aussi
                    if char != "<S>":
                        predicted_line.append(char)
                    else:
                        if self.entities:
                            predicted_line.append(self.entities_dict["remove_space"])
                else:
                    print("Unknown")
                    print(mask)
                    predicted_line.append(char)
            result.append("".join(predicted_line))
            if self.debug:
                print("\n\n--- Début ---\n\n")
                print(sentence_masked_zipped)
                print("".join(predicted_line))
                print("\n\n--- Fin ---\n\n")

        if self.remove_breaks:
            result = "".join(result)

        return result

    def tag_and_detect_lb(self, lines):
        """
        Fonctionnement: on va tagger les lignes fusionnées deux à deux.
        :param hyphens: Gérer les césures à la ligne.
        :param entities: Ajouter des entités xml ?
        :return: a list of tuples ('predicted_line': str, break_line: bool)
        """

        pairs = [lines[n] + lines[n + 1] for n in range(len(lines) - 1)]
        # On annote les lignes fusionnées
        preds_list = []
        lines_to_predict = pairs
        for tokenized_string in self.predict_lines(lines_to_predict):
            preds_list.append(tokenized_string)

        zipped_list = list(zip(lines, preds_list))
        processed_list = []

        for index, (orig, prediction) in enumerate(zipped_list):
            # On supprime les espaces. 8 Caractères doivent être suffisants pour éviter toute ambiguité.
            string_to_match = orig[-20:].replace(" ", "")
            # On utilise une expression régulière pour matcher la limite entre les deux lignes, pour les séparer.
            if self.entities:
                expression = f"(\s|{self.entities_dict['add_space']}|{self.entities_dict['remove_space']})*".join(
                    string_to_match)
            else:
                expression = "\s*".join(string_to_match)
            regexp = re.compile(expression)
            result = regexp.search(prediction)
            try:
                borne_sup = result.span(0)[1]
            except:
                print(f"Original string: |{orig}|")
                print(f"Index: {index}")
                print(f"Predicted string: |{prediction}|")
                print(f"Regex: |{expression}|")
                print(f"Chars in prediction: {list(set([char for char in prediction]))}")
                print("Regexp Critical error")
                exit(0)

            # S'il y a une espace, il n'y a pas de césure
            try:
                line_break = prediction[borne_sup] == " "
                # In case of production of xml entities, we have to search for &rien-esp;:
                if not line_break and self.entities:
                    line_break = prediction[borne_sup:borne_sup + 10] == self.entities_dict['add_space']
                if self.debug:
                    print(f"{prediction[:borne_sup]}|{prediction[borne_sup:]}")
                    print(line_break)
            except IndexError as e:
                print(e)
                print(f"String: |{prediction}|")
                print(f"Regex: |{expression}|")
                print(borne_sup)
                print("LB error")
                exit(0)

            # On se débarrasse de la deuxième partie de la ligne.
            prediction = prediction[:borne_sup]

            # S'il y a une césure, on ajoute le tiret (ou autre marqueur) !
            # On peut aussi créer une pseudo balise XML.

            prediction = re.sub(r"\s+", r" ", prediction)

            processed_list.append((prediction, line_break))

        # Gestion de la dernière ligne
        last_line = lines[-1]
        last_prediction = "".join(self.predict_lines(last_line))
        prediction = re.sub(r"\s+", r" ", last_prediction)
        processed_list.append((prediction, True))

        # We remove any leading spaces
        leading_spaces = re.compile('^\s')
        processed_list = [(re.sub(leading_spaces, "", prediction), line_break) for (prediction, line_break) in
                          processed_list]
        return processed_list

    def lines_to_tensor(self, lines: str):
        '''
        This functions takes a list of lines and converts it to a tensor before prediction
        '''

        max_length = max([len(sentence) for sentence in lines])
        numerical_sentences = []
        formatted_sentences = []
        for line in lines:
            splitted_line = [char for char in line]
            example_length = len(line)
            numeric_sentence = [self.input_vocab["<PAD>"], self.input_vocab["<SOS>"]]
            for char in splitted_line:
                if char != " ":
                    if char in self.input_vocab:
                        numeric_sentence.append(self.input_vocab[char])
                    else:
                        numeric_sentence.append(self.input_vocab["<UNK>"])
                else:
                    numeric_sentence.append(self.input_vocab["<S>"])
            # Let's pad the sentence
            numeric_sentence = numeric_sentence + [self.input_vocab["<EOS>"]] + [self.input_vocab["<PAD>"] for _ in
                                                                                 range(max_length - example_length)]
            numerical_sentences.append(numeric_sentence)
            formatted_sentences.append([self.reverse_input_vocab[idx] for idx in numeric_sentence])
        sentences_as_tensors = utils.tensorize(numerical_sentences)
        return sentences_as_tensors, formatted_sentences
