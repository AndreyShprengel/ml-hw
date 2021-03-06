from csv import DictReader, DictWriter
import argparse
import numpy as np
from numpy import array
import nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'
kTROPE_FIELD = 'trope'
kPAGE_FIELD = 'page'
class Stemmer:
    def __init__(self):
		self.stemmer = nltk.PorterStemmer()
      

    def __call__(self, feature_string):
		feature_string = [str(self.stemmer.stem(word)) for word in feature_string.split(" ")]
		stemmed_string = "" 
		for word in feature_string:
			if word == "death":
				word = "die"
			stemmed_string += word + " "
		
		return stemmed_string
		

class Featurizer:
	def __init__(self):
		self.vectorizer =CountVectorizer( ngram_range = (1,3), preprocessor = Stemmer())

	def train_feature(self, examples):
		return self.vectorizer.fit_transform(examples)

	def test_feature(self, examples):
		return self.vectorizer.transform(examples)

	def show_top10(self, classifier, categories):
		feature_names = np.asarray(self.vectorizer.get_feature_names())
		if len(categories) == 2:
			top10 = np.argsort(classifier.coef_[0])[-10:]
			bottom10 = np.argsort(classifier.coef_[0])[:10]
			print("Pos: %s" % " ".join(feature_names[top10]))
			print("Neg: %s" % " ".join(feature_names[bottom10]))
		else:
			for i, category in enumerate(categories):
				top10 = np.argsort(classifier.coef_[i])[-10:]
				print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='LOGREG classifier options')
	parser.add_argument('--test', type=bool, default=False,
						help="Whether to split training data True/False")
	args = parser.parse_args()

	# Cast to list to keep it all in memory
	train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
	test = list(DictReader(open("../data/spoilers/test.csv", 'r')))
	
	testdata = train[13000:len(train)-1]
	if args.test:
		train = train[0:13000]

	feat = Featurizer()

	labels = []
	for line in train:
		if not line[kTARGET_FIELD] in labels:
			labels.append(line[kTARGET_FIELD])

	print("Label set: %s" % str(labels))
	x_train = feat.train_feature(x[kTEXT_FIELD]  + " " + x[kTROPE_FIELD]   for x in train )
	x_mytest = feat.test_feature(x[kTEXT_FIELD]  + " " + x[kTROPE_FIELD]   for x in testdata)
	x_test = feat.test_feature(x[kTEXT_FIELD] + " " + x[kTROPE_FIELD]   for x in test  )

	y_train = array(list(labels.index(x[kTARGET_FIELD])
						 for x in train))

	print(len(train), len(y_train))
	print(set(y_train))

	# Train classifier
	lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
	lr.fit(x_train, y_train)

	feat.show_top10(lr, labels)

	predictions = lr.predict(x_test)
	predictions2 = lr.predict(x_mytest)
	o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
	o.writeheader()
	for ii, pp in zip([x['id'] for x in test], predictions):
		d = {'id': ii, 'spoiler': labels[pp]}
		o.writerow(d)
	total = 0
	for ii in range(len(testdata)):
		if testdata[ii][kTARGET_FIELD] == labels[predictions2[ii]]:
			#print testdata[ii][kTEXT_FIELD] +  testdata[ii][kTARGET_FIELD]
			total += 1
	acc = float(total)/len(testdata)
	print "acc = " + str(acc)
