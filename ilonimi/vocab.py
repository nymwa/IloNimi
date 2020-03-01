from ilonimi import IloNimi

def main():
	tokenizer = IloNimi()
	for token in tokenizer.vocab:
		print(token)

