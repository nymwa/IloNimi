import sys
from ilonimi import IloNimi

def main():
	tokenizer = IloNimi()
	for line in sys.stdin:
		print(' '.join(tokenizer.bert_tokenize(line)))

