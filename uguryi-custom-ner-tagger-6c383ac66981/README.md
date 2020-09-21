# custom-tagger

## Steps

1. Upload the Deciding Force (2017) corpus.

	
	`$ mkdir df-corpus`
	
	`$ aws s3 cp s3://tagworks.thusly.co/decidingforce/corpus/ ./df-corpus --recursive`
	
	`$ find df-corpus/* -maxdepth 0 -type d | wc -l # See how many folders are under df-corpus`

2. Install Stanford CoreNLP.

	`$ wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip`
	
	`$ unzip stanford-corenlp-full-2018-10-05.zip`
	
3. Install Java (or check that Java is already installed).

	`$ java -version`
	
4. Upload `df-classifier.prop`.
	
5. Run the code in `df-classifier.ipynb`. This script proceeds in the following manner.

	- Download stopwords and wordnet
	- Create lemmatizer
	- Import libraries
	- Define auxiliary functions
	- Define main functions
	- Generate train and test data
		- `train.tsv` and `test.tsv` are generated.
	- Train and test model
		- `custom-tagger.ser.gz` is generated when the model is trained.
		- Test results (i.e., true and predicted labels on the test data) are put into the `test-results` folder as a `tsv` file.
	- Check model performance
	
More information on all of these steps can be found in `df-classifier.ipynb`. Just go to [this link](https://nbviewer.jupyter.org/gist/uguryi/ec61b95fe0b075424701b960463700dc/0302_df-classifier.ipynb) for a nice rendering of this Jupyter notebook.

A `.py` version of this file is included as well. 