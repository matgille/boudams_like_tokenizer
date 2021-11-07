import sys

import trainer as trainer
import utils as utils
import tagger as tagger

# https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb


test_string_normalized = "dieronlebatalla,evencieronleeganarondelcuantoquisieron.Desi\n" \
                         "delrregnadodelrreydonalfonsoquefueenlaera\n" \
                         "ylliricoagrandpriesaEescojodesuhuestea\n" \
                         "eraaquellojusticiacomodereyede\n" \
                         "ensucabodiez\n" \
                         "trayçionquelefezieronlosmorosdetoledoquandorresçibieron\n" \
                         "edixoaellosassi,segundcuentaJosefo:\n" \
                         "nuevoera.PlogoaMercurioconestapregunta,e\n" \
                         "fazieelmuchamaldattanmaloetanfalsofue\n" \
                         "deviera,enonlapuedaperdernindexar\n" \
                         "edeAaron,edecomomurioMoisen"

graphematic_test_string = "E bien asi el cauallero viçioso\n" \
                          "enemigos ante que ellos se apercibã de⸗sta\n" \
                          "de lasethi cas. Ca dezia el manos sõ vnos\n" \
                          "de los mejoꝛesmiẽbꝛos\n" \
                          "el anzuelo de ladiuinid et conq̅\n" \
                          "la moꝛtandat q̅ los caualleros nȯ\n" \
                          "menonfallescadesenlamar. ꝫ\n" \
                          "lapodꝛatanto lançar. Esy lança mu⸗cho\n" \
                          "ferir enalgunt lugar silancan a\n" \
                          "E trae enxemploalli de fel ipo: q̃ embio\n" \
                          "comẽçar la faziendapoꝛ otra parte. E\n" \
                          "enemigodonde muera. la setena\n" \
                          "Edizequemuymejoꝛessõꝑacaualle⸗ria"

train_path = "/home/mgl/Bureau/These/datasets/segmentation_segmentor/datasets/train/train.tsv"
test_path = "/home/mgl/Bureau/These/datasets/segmentation_segmentor/datasets/test/test.tsv"

# train_path = "../.data/train.txt"
# test_path = "../.data/test.txt"
if sys.argv[1] == "train":
    trainer = trainer.Trainer(batch_size=64,
                              epochs=2,
                              lr=0.0005,
                              train_path=train_path,
                              test_path=test_path)
    trainer.train()

elif sys.argv[1] == "tag_xml":
    tagger = tagger.Tagger(device="cpu",
                           input_vocab="../models/input_vocab.voc",
                           target_vocab="../models/target_vocab.voc",
                           model="../models/model_tokenizer.pt",
                           remove_breaks=False,
                           verbosity=False)

    tagger.tokenize_xml(sys.argv[2])



elif sys.argv[1] == "tag_txt":
    tagger = tagger.Tagger(device="cuda:0",
                           input_vocab="../models/input_vocab.voc",
                           target_vocab="../models/target_vocab.voc",
                           model="../models/model_tokenizer.pt",
                           remove_breaks=False,
                           XML_entities=False,
                           debug=False)

    tagger.tokenize_txt(sys.argv[2])