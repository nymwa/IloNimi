import sys
from ilonimi import IloNimi
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--bos', action='store_true')
	parser.add_argument('--mask', action='store_true')
	args = parser.parse_args()

	tokenizer = IloNimi(bos=args.bos, mask=args.mask)
	for line in sys.stdin:
		print(' '.join(tokenizer.bert_tokenize(line)))

