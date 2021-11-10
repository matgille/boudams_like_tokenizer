# Boudams like segmenter

Fork from Thibault Clérice's Boudams tokenizer [Clérice 2020] ([repo](https://github.com/PonteIneptique/boudams)).

See also [Ben Trevett's notebook](https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb)
from which the code is derived. 

## What differs from Boudams

Besides `<SOS>` and `<EOS>`, three classes are predicted, instead of two. This allows to keep the original segmentation 
information:
- Word Boundary > Word Content (remove a space). Example: `fizo<X>los`
- Word Content > Word boundary (add a space). Example: `de<X>las`
- Word Content (do nothing). Example: `la<X>caualleria`

## Training a model

The program expects correctly segmented lines of text as the training data.

Set path to test and train sets in `tokeniser.py` file. 

`python tokeniser.py train`


## Outputs

The tagger supports txt and xml inputs/outputs.
It allows to produce an XML file with entities to easier
indicate the segmentation information.


## References

Clérice, Thibault. “Evaluating Deep Learning Methods for Word Segmentation of Scripta Continua Texts in Old French and Latin.” Journal of Data Mining & Digital Humanities 2020 (April 7, 2020). Accessed February 26, 2021. https://jdmdh.episciences.org/6264/pdf.
