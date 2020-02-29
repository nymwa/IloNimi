import sys
from ilonimi import IloNimi
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='tokenize')
	args = parser.parse_args()
	tokenizer = IloNimi()

	if args.mode == 'tokenize':
		for line in sys.stdin:
			print(' '.join(tokenizer.bert_tokenize(line)))
	elif args.mode == 'encode':
		for line in sys.stdin:
			print(' '.join([str(x) for x in tokenizer.encode(line)]))
	elif args.mode == 'decode':
		for line in sys.stdin:
			print(tokenizer.decode([int(x) for x in line.strip().split(' ')]))
	elif args.mode == 'vocab':
		for token in tokenizer.vocab:
			print(token)

