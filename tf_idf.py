import math

def freq(word, doc):
	return doc.count(word)
 
 
def word_count(doc):
	return len(doc)
 
 
def getTF(word, doc):
	return (freq(word, doc) / float(word_count(doc)))
 
 
def num_docs_containing(word, list_of_docs):
	count = 0
	for document in list_of_docs:
		if freq(word, document) > 0:
			count += 1
	return 1 + count
 
 
def getIDF(word, list_of_docs):
	return math.log(len(list_of_docs) / float(num_docs_containing(word, list_of_docs)))

def return_idf(n_docs_containing_word, n_docs):
	return math.log(n_docs / float(n_docs_containing_word))
 