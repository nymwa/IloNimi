import sys
from ilonimi import IloNimi

if __name__ == '__main__':
	tokenizer = IloNimi()
	for line in sys.stdin:
		tokenizer.show(line)

