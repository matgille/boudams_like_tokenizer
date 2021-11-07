# Boudams like segmenter

Fork from Thibault Clérice's Boudams tokenizer [Clérice 2020] ([repo](https://github.com/PonteIneptique/boudams)), rebuilt taking in 
considerations the conclusions of his paper on the best architecture for this task.

See also [Ben Trevett's notebook](https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb)
from which the code is derived. 

## What differs from Boudams

Three classes are predicted, instead of two. This allows to keep the original segmentation 
information:
- Word Boundary > Word Content (remove a space). Example: `fizo los`
- Word Content > Word boundary (add a space). Example: `delas`
- Word Content (do nothing). Example: `la caualleria`

## Outputs

The tagger supports txt and xml inputs/outputs.
It allows to produce an XML file with entities to easier
indicate the segmentation information.


## References

Clérice, Thibault. “Evaluating Deep Learning Methods for Word Segmentation of Scripta Continua Texts in Old French and Latin.” Journal of Data Mining & Digital Humanities 2020 (April 7, 2020). Accessed February 26, 2021. https://jdmdh.episciences.org/6264/pdf.
