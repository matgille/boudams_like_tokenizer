# Boudams like segmenter

Fork from Thibault Clérice's Boudams tokenizer [Clérice 2020] ([repo](https://github.com/PonteIneptique/boudams)). 

See also [Ben Trevett's notebook](https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb)
from which part of the code is derived. 

## What differs from Boudams

Besides `<PAD>`, `<SOS>` and `<EOS>`, four classes are predicted, instead of two. This allows to keep the original segmentation 
information:
- Word Boundary > Word Content (remove a space). Class `<S-s>` Example: `fizo< >los`
- Word Content > Word boundary (add a space). Class  `<s-S>` Example: `d<e>las`
- Word Content (do nothing). Class `<S-S>` Example: `caua<u>lleria`
- Space (do nothing). Class `<s-s>` Example: `la< >cauallería`

## Training a model

The program expects correctly segmented lines of text (with line breaks) as the training data.

Set path to test and train sets in `tokeniser.py` file. 

`python tokeniser.py train`


## Tagging text

The tagger supports txt and xml-tei inputs/outputs (as long as they contain `tei:lb` elements that mark line begginings).
It allows to produce an XML file with entities to easier
indicate the segmentation information [Stutzmann 2013, Pinche 2017].


## References

- Clérice, Thibault. “Evaluating Deep Learning Methods for Word Segmentation of Scripta Continua Texts in Old French and Latin.” Journal of Data Mining & Digital Humanities 2020 (April 7, 2020). Accessed February 26, 2021. https://jdmdh.episciences.org/6264/pdf.
- Pinche, Ariane. “Édition Nativement Numérique Des Oeuvres Hagiographiques ‘Li Seint Confessor’ de Wauchier de Denain, d’après Le Manuscrit 412 de La Bibliothèque Nationale de France,” 2017. Accessed November 10, 2021. https://hal.archives-ouvertes.fr/hal-01628533.
- Stutzmann, Dominique. “Ontologie des formes et encodage des textes manuscrits médiévaux.” Document numérique Vol. 16, no. 3 (2013): 81–95.
