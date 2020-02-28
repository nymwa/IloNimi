import sys
import re
import numpy as np
import string
from collections import Counter

class IloNimi:
	instance = None
	def __new__(cls):
		if cls.instance is None:
			cls.instance = super().__new__(cls)
			punctuations = '!"#\$%&\'\(\)\*\+,-\.\/:;<=>\?@\[\\\\\]\^_`{\|}~'
			cls.tag_list = '[PAD] [UNK] [CLS] [SEP] [MASK]'.split(' ')
			cls.punct_list = list(string.punctuation)
			cls.digits = list('0123456789')
			cls.word_list = 'a akesi ala alasa ale ali anpa ante anu apeja awen e en esun ijo ike iki ilo insa jaki jan jelo jo kala kalama kama kan kapa kapesi kasi ken kepeken kijetesantakalu kili kin kipisi kiwen ko kon kule kulupu kute la lape laso lawa leko len lete li lili linja lipu loje lon luka lukin lupa ma majuna mama mani meli mi mije moku moli monsi monsuta mu mulapisu mun musi mute namako nanpa nasa nasin nena ni nimi noka o oko olin ona open pakala pake pali palisa pan pana pasila pata pi pilin pimeja pini pipi po poka poki pona powe pu sama seli selo seme sewi sijelo sike sin sina sinpin sitelen sona soweli suli suno supa suwi tan taso tawa telo tenpo toki tomo tu tuli unpa uta utala walo wan waso wawa weka wile yupekosi'.split(' ')
			vowel = ['A', 'E', 'I', 'O', 'U']
			consonant = ['K', 'L', 'M', 'N', 'P', 'S', 'T', 'W', 'J']
			cls.syll_list = vowel.copy()
			for c in consonant:
				for v in vowel:
					if c + v not in ['TI', 'WO', 'WU', 'JI']:
						cls.syll_list.append(c + v)
						cls.syll_list.append('@@' + c + v)
			cls.syll_list.append('@@N')
			# vocabulary
			cls.vocab_list = cls.tag_list + cls.punct_list + cls.word_list
			cls.vocab_set = set(cls.vocab_list)
			# regex
			cls.left_punct_pattern  = re.compile(r'(?<=[^\s])([{}])'.format(punctuations, punctuations))
			cls.right_punct_pattern = re.compile(r'([{}])(?=[^\s])'.format(punctuations, punctuations))
			cls.space_pattern = re.compile(r'\s+')
			cls.proper_pattern = re.compile(r'^([AIUEO]|[KSNPML][aiueo]|[TJ][aueo]|W[aie])n?(([ksnpml][aiueo]|[tj][aueo]|w[aie])n?)*$')
		return cls.instance

	def preprocess(self, x):
		x = x.translate(str.maketrans({'„':'"', '“':'"', '”':'"', '–':'-'}))
		return x

	def spell_correction(self, x):
		dct = {'aksei':'akesi', 'epelanto':'Epelanto', 'Kanze':'Kanse', 'Maria':'Malia', 'pimjea':'pimeja', 'Pulowo':'Pulo', 'siejelo':'sijelo', 'siejlo':'sijelo', 'sielo':'sijelo', 'sitele':'sitelen', 'sitelten':'sitelen', 'sitlen':'sitelen', 'tempo':'tenpo', 'tenp':'tenpo', 'Tatoeba':'Tatojepa', 'Tom':'Ton', 'utalta':'utala'}
		return dct[x] if x in dct else x

	def convert_unk(self, x):
		x = [w if w in self.vocab_set or self.is_proper(w) or w.isdecimal() else '[UNK]' for w in x]
		return x

	def is_proper(self, x):
		return self.proper_pattern.match(x)

	def split_syllables(self, x):
		re_syll = re.compile(r'[ksnpmltjw][aiueo]n?')
		lst, sub = [], ''
		for i in range(len(x))[::-1]:
			sub, x = x[-1] + sub, x[:-1]
			if re_syll.match(sub):
				lst, sub = [sub] + lst, ''
		return lst if sub == '' else [sub] + lst

	def split_punct(self, x):
		x = self.left_punct_pattern.sub(' \\1', x)
		x = self.right_punct_pattern.sub('\\1 ', x)
		return x

	def category(self, x):
		if x in self.tag_list:
			return 'tag'
		elif x in self.punct_list:
			return 'punct'
		elif x in self.vocab_list:
			return 'word'
		elif self.is_proper(x):
			return 'proper'
		elif all(c in self.digits for c in x):
			return 'number'
		else:
			return 'unk'

	def show(self, x, spell_check=True):
		lst = self.__call__(x, spell_check)
		print('S {}'.format(' '.join([dct['token'] for dct in lst])))
		for dct in lst:
			out = 'W {}\t{}'.format(dct['token'], dct['category'])
			print(out)
		print('EOS\n')

	def __call__(self, x, spell_check):
		x = x.strip()
		x = self.preprocess(x)
		x = self.split_punct(x)
		x = self.space_pattern.sub(x, ' ')
		x = x.split(' ')
		lst = []
		for w in x:
			if spell_check:
				w = self.spell_correction(w)
			dct = {'token':w, 'category':self.category(w)}
			lst.append(dct)
		return lst


class IloNimiBERT(IloNimi):
	def __init__(self):
		super().__init__()
		self.bert_vocab = self.tag_list + self.punct_list + self.digits + self.word_list + self.syll_list

	def split_proper_for_bert(self, x):
		re_cv = re.compile(r'[ksnpmltjw][aiueo]')
		lst, sub = [], ''
		for i in range(len(x))[::-1]:
			sub = x[-1] + sub
			x = x[:-1]
			if sub == 'n' or re_cv.match(sub):
				lst = ['@@' + sub.upper()] + lst
				sub = ''
		if sub != '':
			lst = [sub.upper()] + lst
		return lst

	def encode(self, x, rm_unk=False, spell_check=True):
		lst = self.__call__(x, spell_check)
		tmp = ['[CLS]']
		for dct in lst:
			if dct['category'] == 'proper':
				tmp += [x.upper() for x in self.split_proper_for_bert(dct['token'])]
			elif dct['category'] == 'number':
				tmp += [x for x in list(dct['token'])]
			elif dct['category'] == 'unk':
				tmp.append('[UNK]')
			else:
				tmp.append(dct['token'])
		tmp.append('[SEP]')
		return tmp

	def decode(self, x, tag_word='≡╹ω╹≡'):
		x = [self.bert_vocab[n] for n in x[1:-1]]
		x = [(tag_word if (w in self.tag_list) else w) for w in x]
		# for proper noun
		for i in range(len(x)):
			if x[i].isupper():
				x[i] = x[i].lower() if x[i].startswith('@@') else x[i].capitalize()
		x = ' '.join(x)
		x = x.replace(' @@', '')
		x = re.sub(r'(\w) ([^\w≡])', r'\1\2', x)
		return x

