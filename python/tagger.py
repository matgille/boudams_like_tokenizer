import re
import unicodedata

import lxml.etree as etree

import torch
import numpy as np
import tqdm

import utils as utils


# TODO: add an option to output break separated value.


class Tagger:
    def __init__(self, device, input_vocab: str, target_vocab: str, model: str, remove_breaks: bool, XML_entities: bool, debug:bool):
        self.device = device
        if self.device == 'cpu':
            self.model = torch.load(model, map_location=self.device)
        else:
            self.model = torch.load(model).to(self.device)
        self.input_vocab = torch.load(input_vocab)
        self.target_vocab = torch.load(target_vocab)
        self.reverse_input_vocab = {v: k for k, v in self.input_vocab.items()}
        self.reverse_target_vocab = {v: k for k, v in self.target_vocab.items()}
        self.remove_breaks = remove_breaks  # Keep the linebreaks in prediction?
        self.entities = XML_entities  # Whether to display original segmentation information
        self.debug = debug

    def tokenize_xml(self, xml_file):
        tei_namespace = 'http://www.tei-c.org/ns/1.0'
        namespace_declaration = {'tei': tei_namespace}
        with open(xml_file, "r") as input_file:
            f = etree.parse(input_file)
        line_breaks = f.xpath("//tei:lb[not(parent::tei:fw)]", namespaces=namespace_declaration)
        text_lines = [utils.clean_and_normalize_encoding(line.tail) for line in line_breaks]

        predicted = self.tag_and_detect_lb(text_lines)
        print(predicted)

        zipped = list(zip(line_breaks, predicted))

        for xml_element, (text, lb) in zipped:
            if lb == True:
                xml_element.set("break", "n")
            else:
                xml_element.set("break", "y")
            xml_element.tail = text

        with open(xml_file.replace(".xml", ".tokenized.xml"), "w") as output_file:
            output_file.write(etree.tostring(f, pretty_print=True).decode())

    def tokenize_txt(self, txt_file):
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
            result = []
            batch_size = 256
            steps = len(text_lines) // 256
            for n in tqdm.tqdm(range(steps)):
                fragment = text_lines[n*steps: n*steps + batch_size]
                predictions = self.tag_and_detect_lb(fragment)
                result.extend(predictions)
        else:
            result = self.tag_and_detect_lb(text_lines)
        result = [f'{text}-' if line_break is True else text for (text, line_break) in result]
        with open(txt_file.replace('.txt', '.tokenized.txt'), "w") as input_file:
            input_file.write("\n".join(result))
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
            # print(sentence)
            # print(predict_mask)
            # exit(0)
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
                elif mask == "<WB>":
                    if char != "<S>":
                        predicted_line.append(char)
                    else:
                        predicted_line.append(" ")
                    if self.entities:
                        predicted_line.append("&rien-esp;")
                    else:
                        predicted_line.append(" ")
                elif mask == "<s-S>":
                    # Ici aussi
                    if char != "<S>":
                        predicted_line.append(char)
                    else:
                        if self.entities:
                            predicted_line.append("&esp-rien;")
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
                line_break = prediction[borne_sup] is not " "
            except:
                print(f"Index: {index}")
                print(f"String: |{prediction}|")
                print(f"Regex: |{expression}|")
                print(borne_sup)
                print("LB Critical error")
                exit(0)

            # On se débarrasse de la deuxième partie de la ligne.
            prediction = prediction[:borne_sup]

            # S'il y a une césure, on ajoute le tiret (ou autre marqueur) !
            # On peut aussi créer une pseudo balise XML.

            prediction = re.sub(r"\s+", r" ", prediction)
            prediction = (prediction, line_break)

            processed_list.append(prediction)

        # Gestion de la dernière ligne
        last_prediction, last_string = zipped_list[-1]
        result = regexp.search(last_prediction)
        borne_sup = result.span(0)[1]
        prediction = last_prediction[borne_sup:]
        prediction = re.sub(r"\s+", r" ", prediction)
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
