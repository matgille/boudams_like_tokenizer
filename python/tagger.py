import torch
import numpy as np
import python.utils as utils
import python.seq2seq as seq2seq
import python.model as model


class Tagger:
    def __init__(self, input_vocab: str, target_vocab: str, model: str, verbosity: bool):
        self.model = torch.load(model).to("cuda:0")
        self.input_vocab = torch.load(input_vocab)
        self.target_vocab = torch.load(target_vocab)
        self.reverse_input_vocab = {v: k for k, v in self.input_vocab.items()}
        self.reverse_target_vocab = {v: k for k, v in self.target_vocab.items()}
        self.verbose = verbosity  # Whether to display original segmentation information

    def predict(self, sentences: str) -> str:
        input_tensor, formatted_inputs = self.sentence_to_tensor(sentences)
        input_size = len(formatted_inputs)
        input_tensor = input_tensor.to("cuda:0")
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
            print(formatted_inputs[sentence_number])
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
            final_pred = []
            for char, mask in sentence_masked_zipped:
                if mask == "<WC>":
                    final_pred.append(char)
                elif mask == "<s-s>":
                    final_pred.append(" ")
                elif mask == "<WB>":
                    final_pred.append(char)
                    if self.verbose:
                        final_pred.append("&rien-esp;")
                    else:
                        final_pred.append(" ")
                elif mask == "<s-S>":
                    if self.verbose:
                        final_pred.append("&esp-rien;")

            final_pred = "".join(final_pred)
            result.append(final_pred)
        print(f"Prediction: {result}")
        return "\n".join(result)

    def sentence_to_tensor(self, sentence):
        max_length = max([len(sentence) for sentence in sentence.split("\n")])
        numerical_sentences = []
        formatted_sentences = []
        for sentence in sentence.split("\n"):
            splitted_sentence = [char for char in sentence]
            example_length = len(sentence)
            numeric_sentence = [self.input_vocab["<PAD>"], self.input_vocab["<SOS>"]]
            for char in splitted_sentence:
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
