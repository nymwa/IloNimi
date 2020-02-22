import sys
from ilonimi import IloNimi
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='default')
	parser.add_argument('--vocab', action='store_true')
	args = parser.parse_args()

	tokenizer = IloNimi()
	if args.vocab:
		for token in tokenizer.bert_vocab():
			print(token)
	else:
		if args.mode == 'default':
			for line in sys.stdin:
				tokenizer.show(line)
		elif args.mode == 'bert':
			for line in sys.stdin:
				tokenizer.show_bert(line)

