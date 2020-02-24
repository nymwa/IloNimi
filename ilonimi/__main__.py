import sys
from ilonimi import IloNimi, IloNimiBERT
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='default')
	parser.add_argument('--vocab', action='store_true')
	args = parser.parse_args()

	if args.vocab:
		tokenizer = IloNimiBERT()
	else:
		if args.mode == 'default':
			tokenizer = IloNimi()
		elif args.mode == 'bert':
			tokenizer = IloNimiBERT()

	if args.vocab:
		for token in tokenizer.bert_vocab:
			print(token)
	else:
		if args.mode == 'default':
			for line in sys.stdin:
				tokenizer.show(line)
		elif args.mode == 'bert':
			for line in sys.stdin:
				print(' '.join(tokenizer.encode(line)))

