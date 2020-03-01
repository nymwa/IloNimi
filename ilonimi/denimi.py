import sys
from ilonimi import IloNimi

def main():
	tokenizer = IloNimi()
	for line in sys.stdin:
		line = line.strip()
		if line.startswith('[CLS]'):
			line = line[5:]
		if line.endswith('[SEP]'):
			line = line[:-5]
		print(tokenizer.bert_detokenize(line))

