import os
import sys
import re
import numpy as np
import string
from collections import Counter

class SpellChecker:
	def __init__(self):
		with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'correction.tsv')) as f:
			lst = [line.split('\t') for line in f.read().splitlines()]
		self.dct = {e:c for e, c in lst}

	def __call__(self, x):
		if x in self.dct:
			x = self.dct[x]
		return x


class Normalizer:
	def __init__(self):
		self.special_punct = str.maketrans({'„':'"', '“':'"', '”':'"', '–':'-', '▁':'_'})
		self.left_punct  = re.compile(r'(?<=[^\s])([\W_])')
		self.right_punct = re.compile(r'([\W_])(?=[^\s])')
		self.number  = re.compile(r'(\d+)')
		self.space = re.compile(r'\s+')

	def __call__(self, x):
		x = x.strip()
		x = x.translate(self.special_punct)
		x = self.left_punct.sub(r' \1', x)
		x = self.right_punct.sub(r'\1 ', x)
		x = self.number.sub(r' \1 ', x)
		x = self.space.sub(' ', x)
		return x


class ProperSplitter:
	def __init__(self):
		self.proper = re.compile(r'^([AIUEO]|[KSNPML][aiueo]|[TJ][aueo]|W[aie])n?(([ksnpml][aiueo]|[tj][aueo]|w[aie])n?)*$')
		self.cv = re.compile(r'[ksnpmltjw][aiueo]')
	
	def gen_syllables(self):
		vowels = ['A', 'E', 'I', 'O', 'U']
		consonants = ['K', 'L', 'M', 'N', 'P', 'S', 'T', 'W', 'J']
		return vowels + [t for c in consonants for v in vowels for t in [c+v, '▁'+c+v] if c+v not in ['TI', 'WO', 'WU', 'JI']] + ['▁N']

	def check(self, x):
		return self.proper.match(x)

	def __call__(self, x):
		assert self.check(x)
		if x[-1] == 'n':
			return self.__call__(x[:-1]) + ['▁N']
		elif re.search('[ksnpmltjw][aiueo]$', x):
			return self.__call__(x[:-2]) + ['▁' + x[-2:].upper()]
		else:
			return [x.upper()]


class NumberSplitter:
	def gen_digits(self):
		return [t for n in range(10) for t in [str(n), str(n)+'▁']]
	
	def __call__(self, x):
		assert x.isdecimal()
		x = str(int(x))
		head, last = x[:-1], x[-1]
		return [n+'▁' for n in head] + [last]


class IloNimi:
	def __init__(self):
		self.spell_checker = SpellChecker()
		self.normalizer = Normalizer()
		self.number_splitter = NumberSplitter()
		self.proper_splitter = ProperSplitter()
		self.prepare_vocab()
		self.vocab_dict = {x:i for i, x in enumerate(self.vocab)}

	def prepare_vocab(self): 
		# special tokens
		self.special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
		self.vocab = self.special_tokens.copy()

		# punctuations
		self.vocab += list(string.punctuation)

		# digits
		self.vocab += self.number_splitter.gen_digits()

		# words
		with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'wordlist.txt')) as f:
			self.vocab += f.read().splitlines()

		# proper nouns
		self.vocab += self.proper_splitter.gen_syllables()

	def tokenize(self, x):
		x = self.normalizer(x)
		x = x.split(' ')
		lst = []
		for t in x:
			t = self.spell_checker(t)
			if self.proper_splitter.check(t):
				lst += self.proper_splitter(t)
			elif t.isdecimal():
				lst += self.number_splitter(t)
			else:
				lst.append(t)
		return lst

	def bert_tokenize(self, x):
		return ['[CLS]'] + self.tokenize(x) + ['[SEP]']

	def encode(self, x):
		x = self.tokenize(x)
		x = ['[CLS]'] + x + ['[SEP]']
		return [self.vocab_dict[t if t in self.vocab_dict else '[UNK]'] for t in x]

	def decode(self, x, unk='≡╹ω╹≡'):
		x = [self.vocab[t] for t in x]
		if x[0] == '[CLS]':
			x = x[1:]
		if x[-1] == '[SEP]':
			x = x[:-1]
		x = [unk if t in self.special_tokens else t for t in x]
		x = ' '.join(x)
		x = re.sub(r'(?<=[A-Z]) ▁(?=[A-Z]+)', '', x)
		x = re.sub(r'(?<=\d)▁ (?=\d)', '', x)
		x = re.sub(r'▁', '', x)
		x = re.sub(r'[A-Z]+', lambda x: x.group().capitalize(), x)
		return x

