import os
import random
import sys

import numpy as np


def main(corpus, proportions, max_example_length):
	with open(corpus, 'r') as corpus_file:
		corpus_file_as_list = [example.replace("\n", "") for example in corpus_file.readlines()]

	# On va réduire la taille des exemples d'abord
	corpus_to_split = []
	for item in corpus_file_as_list:
		splitted = item.split()
		length = len(splitted)
		if length > max_example_length:
			splits = [" ".join(splitted[i:i + max_example_length]) for i in range(0, len(splitted), max_example_length)]
			corpus_to_split.extend(splits)
		else:
			corpus_to_split.append(item)
	assert np.max([len(item.split()) for item in corpus_to_split]) == max_example_length, "Max example length issue, not matching argument"
	assert len(corpus_to_split) > len(corpus_file_as_list), "List with reduced examples must be longer than original list."

	# Puis on crée les corpus
	random.shuffle(corpus_to_split)
	test_corpus = []
	train_corpus = []
	dev_corpus = []

	proportions['test'] = round(proportions['test'] * len(corpus_to_split))
	proportions['dev'] = round(proportions['dev'] * len(corpus_to_split))
	proportions['train'] = round(proportions['train'] * len(corpus_to_split))

	test_corpus.extend(corpus_to_split[:proportions['test']])
	train_corpus.extend(corpus_to_split[proportions['test']:proportions['test'] + proportions['train']])
	dev_corpus.extend(corpus_to_split[proportions['test'] + proportions['train']:])

	input_file_dir = "/".join(corpus.split("/")[:-1]) + "/"
	try:
		os.makedirs(f"{input_file_dir}/test")
	except FileExistsError:
		pass
	with open(f"{input_file_dir}/test/test.txt", "w") as test_file:
		test_file.write("\n".join(test_corpus))
	try:
		os.makedirs(f"{input_file_dir}/train")
	except FileExistsError:
		pass
	with open(f"{input_file_dir}/train/train.txt", "w") as train_file:
		train_file.write("\n".join(train_corpus))
	try:
		os.makedirs(f"{input_file_dir}/dev")
	except FileExistsError:
		pass
	with open(f"{input_file_dir}/dev/dev.txt", "w") as dev_file:
		dev_file.write("\n".join(dev_corpus))


if __name__ == '__main__':
	input_corpus = sys.argv[1]
	proportions = {'train': .9, 'test': .1, 'dev': 0}
	max_example_length = int(sys.argv[2])
	main(input_corpus, proportions, max_example_length)