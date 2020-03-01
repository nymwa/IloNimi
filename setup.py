import setuptools
 
setuptools.setup(
	name="ilonimi",
	version="1.0",
	author="nymwa",
	author_email="nymwa0@gmail.com",
	description="Toki Pona Tokenizer",
	packages=setuptools.find_packages(),
	entry_points={
		'console_scripts':[
				'ennimi = ilonimi.ennimi:main',
				'denimi = ilonimi.denimi:main',
				'nimiale = ilonimi.vocab:main',
				'nimiali = ilonimi.vocab:main',
			]},
	package_data={'ilonimi':['correction.tsv', 'wordlist.txt']},
	classifiers=["Programming Language :: Python :: 3.7"],
)
