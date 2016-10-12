import pickle
import pandas as pd
import datetime
import tf_idf
import operator
from datetime import date
from similarity_metrics import *
from math import *
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import PorterStemmer, WordPunctTokenizer

# A global list to store mapping of tf-idf values to feature names
feature_names = []
# A global list to store tf-idf values for tweets text
tfidf_titles = []

def corpus_list(tweets_dict, feature_index, sample_num):
	"""
	Args: tweets_dict -> data dictionary extracted, keys from 0 to sample_num - 1, 
						 values = list of feature data
		  feature_index -> Index of feature to be accessed in the dictionary.
	Return sequence of strings for corpus
	Length of sequence = sample_num 
	"""
	string_list = []
	for snum in xrange(0, sample_num):
		string_list.append(tweets_dict[snum][feature_index])
	return string_list

def clean_doc(doc):
	""" Lemmatizing and Tokenization for text data/ doc. Arg: doc -> string"""
	# stemmer = PorterStemmer()
	lmtzr = WordNetLemmatizer()
	tokens = WordPunctTokenizer().tokenize(doc.decode('utf-8'))
	clean = [token.lower() for token in tokens if token.lower() and len(token) > 2]
	# final = [stemmer.stem(word) for word in clean]
	final = [lmtzr.lemmatize(word, pos = 'v') for word in clean]
	final_doc = ""
	for string in final:
		final_doc += string + " "
	return final_doc

def clean_docs(feature_docs):
	""" 
	For a given feature, feature docs arg has all docs corresp. to that feature 
	We pass these docs to clean_doc function for stemming and tokenization of textual data
	Args: feature_docs -> list of strings
	"""
	idx = 0
	for doc in feature_docs:
		feature_docs[idx] = clean_doc(doc)
		idx += 1

def gen_tfidf_array(cleaned_doc_list):
	"""Generate and return tf-idf vector array from list of textual documents"""
	global feature_names
	vectorizer = TfidfVectorizer(min_df=1, stop_words = 'english')
	Y = vectorizer.fit_transform(cleaned_doc_list)
	feature_names = vectorizer.get_feature_names()
	Y = Y.toarray()
	return Y	

def get_days_list(data_dict, date_feature_index):
	base_day = date(1970, 1, 1)
	days_list = []
	for snum in data_dict:
		# days_list.append((data_dict[snum][date_feature_index] - base_day).days)
		days_list.append((data_dict[snum][date_feature_index] - base_day).days * 1440 + (data_dict[snum][date_feature_index] - base_day).seconds)
	return days_list

def get_location_list(data_dict, location_feature_index1, location_feature_index2):
	location_list = []
	for snum in data_dict:
		location_list.append([data_dict[snum][location_feature_index1], data_dict[snum][location_feature_index2]])
	return location_list


class final_cluster:
	"""
	This is cluster class for the final clusters generated after binary weighted voting
	Data Members:
	data_points_idx : list of indices of data points in the cluster
	num_clusters : static -> Keeping count of the number of clusters formed
	"""
	num_clusters = 0
	def __init__(self, data_point_idx, data_point):
		self.data_points_idx = [data_point_idx]
		self.num_points = 1
		self.centroid = [x for x in data_point]
		self.sum_points = [x for x in data_point]
		final_cluster.num_clusters += 1

	def score(self, data_point, weight, pred_lists):
		score = 0.0
		count = 0
		for idx in self.data_points_idx:
			count += len(pred_lists)
			for feature_index in xrange(0, len(pred_lists)):  		
				if pred_lists[feature_index][data_point] == pred_lists[feature_index][idx]:
					score += weight[feature_index]
		return score/float(count)

	def add_point(self, data_point_idx, data_point):
		self.num_points += 1	
		self.sum_points = [sum(x) for x in zip(self.sum_points, data_point)]
		self.centroid = [x/float(self.num_points) for x in self.sum_points]
		self.data_points_idx.append(data_point_idx)


class cluster:
	cluster_type_list = ['title', 'description', 'date', 'location']
	num_clusters = [0]*4
	# To be used only when cluster_type_num = 4 

	def __init__(self, cluster_type_num, centroid, data_point_idx):
		self.num_points = 1
		self.cluster_type_num = cluster_type_num
		cluster.num_clusters[cluster_type_num] += 1
		self.point_indices = [data_point_idx]
		if cluster_type_num == 0:
			self.sum_points = [x for x in centroid]
			self.centroid = [x for x in centroid]
		else:
			self.sum_points = centroid
			self.centroid = centroid
		# self.cluster_type = cluster_type_list[cluster_type_num]

	def similarity(self, data_point):
		if self.cluster_type_num in [0,1]:
			return cosine_sim_metric(self.centroid, data_point)
		elif self.cluster_type_num == 2:
			return date_similarity_metric(self.centroid, data_point)

	def add_point(self, data_point_metric, data_point_idx):
		if self.cluster_type_num in [0,1]:
			self.sum_points = [sum(x) for x in zip(self.sum_points, data_point_metric)]
			self.centroid = [x/float(self.num_points + 1) for x in self.sum_points] 
		elif self.cluster_type_num == 2:
			self.sum_points += data_point_metric
			self.centroid = self.sum_points/float(self.num_points + 1)
		self.point_indices.append(data_point_idx)
		self.num_points += 1
# final_cluster Class End


def find_MBS(data_points, point_indices, centroid, cluster_type_num):
	"""Function to calculate MBS for a cluster based on Sumblr paper in Sigr 2013"""
	sum_sim = 0.0
	num = len(point_indices)
	if cluster_type_num in [0,1]:
		for idx in point_indices:
			sum_sim += cosine_sim_metric(centroid, data_points[idx])
	else:
		for idx in point_indices:
			sum_sim += date_similarity_metric(centroid, data_points[idx])
	return sum_sim / float(num)

def clustering_algo(cluster_list, cluster_type_num, data_points):
	""" 
	Algorithm to cluster data points based on individual features indexed by  cluster_type_num 
	cluster_l
	"""
	num_clusters = 0
	idx = 0
	label_pred = []												# Predicted cluster labels for data points
	prev_MBS_update = []
	MBS = []
	if cluster_type_num == 0:
		Beta = 0.10													# Tuned parameter for threshold calculation
	else:
		Beta = 0.70
	for point in data_points:
		if cluster.num_clusters[cluster_type_num] == 0:
			print 'New cluster'
			# Initial MBS value is 1
			MBS.append(1)
			# Previous MBS upadte value is 0 for a new cluster
			prev_MBS_update.append(0)
			first_cluster = cluster(cluster_type_num, point, idx)
			cluster_list.append(first_cluster)					# Maintaining list of all clusters	 	
			num_clusters += 1
			label_pred.append(0)								
		else:
			max_sim = -1										# Parameter to measure max similarity between data point and a cluster centroid
			cnum = 0											# Parameter to measure current cluster number
			maxcnum = 0											# Parameter to measure cluster with max similarity
			for cluster_ in cluster_list:
				sim = cluster_.similarity(point) 
				if sim > max_sim:
					max_sim = sim
					maxcnum = cnum
				cnum += 1
			# Here we have to find MBS
			threshold = Beta*MBS[maxcnum]

			if max_sim >= threshold:
				cluster_list[maxcnum].add_point(point, idx)
				label_pred.append(maxcnum)
				if cluster_list[maxcnum].num_points <= 25:
					MBS[maxcnum] = find_MBS(data_points, cluster_list[maxcnum].point_indices, cluster_list[maxcnum].centroid
									, cluster_type_num)
				elif prev_MBS_update[maxcnum] >= 20:
					prev_MBS_update[maxcnum] = 0
					MBS[maxcnum] = find_MBS(data_points, cluster_list[maxcnum].point_indices, cluster_list[maxcnum].centroid
								, cluster_type_num)
				else:
					prev_MBS_update[maxcnum] += 1
					if cluster_type_num == 0:
						MBS[maxcnum] = (MBS[maxcnum]*(cluster_list[maxcnum].num_points - 1) + 
										cosine_sim_metric(cluster_list[maxcnum].centroid, point))/float(cluster_list[maxcnum].num_points)
					else:
						MBS[maxcnum] = (MBS[maxcnum]*(cluster_list[maxcnum].num_points - 1) + 
										date_similarity_metric(cluster_list[maxcnum].centroid, point))/float(cluster_list[maxcnum].num_points)
			else:
				print 'Adding cluster'
				# Initial MBS value is 1
				MBS.append(1)
				# Previous MBS upadte value is 0 for a new cluster
				prev_MBS_update.append(0)
				new_cluster = cluster(cluster_type_num, point, idx)
				cluster_list.append(new_cluster)
				label_pred.append(cnum)
		idx += 1
	return label_pred 

def super_clustering_algo(cluster_list, sample_num, feature_num, weight, pred_lists):
	"""The final clustering based on binary weighted votes from each feature for a data point"""
	final_label_pred_list = []
	for data_point in xrange(0,sample_num):
		if final_cluster.num_clusters == 0:
			first_final_cluster = final_cluster(data_point, tfidf_titles[data_point])
			cluster_list.append(first_final_cluster)
			final_label_pred_list.append(0)
		else:
			max_score = -1.0
			cnum = 0
			ideal_cnum = 0
			for cluster_ in cluster_list:
				score = cluster_.score(data_point, weight, pred_lists)
				if score > max_score:
					ideal_cnum = cnum
					max_score = score
				cnum += 1
			# Here, we use majority voting and take threshold as 0.5
			threshold = 0.5
			if max_score >= threshold:
				cluster_list[ideal_cnum].add_point(data_point, tfidf_titles[data_point])
				final_label_pred_list.append(ideal_cnum)
			else:
				new_cluster = final_cluster(data_point,tfidf_titles[data_point])
				cluster_list.append(new_cluster)
				final_label_pred_list.append(cnum)

	min_tweets_per_cluster = floor(sqrt(sample_num / len(cluster_list)))
	for fcl in cluster_list:
		if fcl.num_points < min_tweets_per_cluster:
			cluster_list.remove(fcl)
	
	return final_label_pred_list

def generate_all_label_pred(tweets_dict):
	"""Returns prediction labels for all features in a list"""   
	return [generate_title_label_pred(tweets_dict), #generate_description_label_pred(tweets_dict),
			 generate_date_label_pred(tweets_dict)]


def generate_title_label_pred(tweets_dict):
	"""Generate predicted cluster labels for data points based on title similarity"""
	global tfidf_titles
	# Get the list of sample_num number of titles from the dict corpus
	title_list = corpus_list(tweets_dict, 0, sample_num)
	# Taking stems and synonyms into account for string type records
	clean_docs(title_list)
	# Generating tf-idf data for titles
	tfidf_titles = gen_tfidf_array(title_list)
	# List to store cluster objects, based on similarity function for titles
	print '**********************Clusters based on feature = Text*************************'
	ideal_cluster_list = []
	# Getting prediction label (cluster number assigned) for each data point in titles data
	title_label_pred = clustering_algo(ideal_cluster_list, 0, tfidf_titles)
	print 'Number of Clusters = ',
	print len(ideal_cluster_list)
	return title_label_pred


def generate_description_label_pred(tweets_dict):
	"""Generate predicted cluster labels for data points based on description similarity"""
	cluster_list_descriptions = []
	description_list = corpus_list(tweets_dict, 1, sample_num)
	clean_docs(description_list)
	# Generating tf-idf data for descriptions
	tfidf_descriptions = gen_tfidf_array(description_list)
	ideal_cluster_list = []
	# Getting prediction label (cluster number assigned) for each data point in titles data
	description_label_pred = clustering_algo(ideal_cluster_list, 1, tfidf_descriptions)

	print '**********************Clusters based on feature = Description*************************'
	print 'Predicted Cluster Labelling = ',
	print description_label_pred
	print 'Number of Clusters = ',
	print len(ideal_cluster_list)
	return description_label_pred


def generate_date_label_pred(tweets_dict):
	"""Generate predicted cluster labels for data points based on description similarity"""
	days_list = get_days_list(tweets_dict, 1)
	difference = max(days_list) - min(days_list)
	print '**********************Clusters based on feature = DateTime*************************'
	cluster_list_dates = []
	ideal_cluster_list = []
	# Getting prediction label (cluster number assigned) for each data point in date data
	date_label_pred = clustering_algo(ideal_cluster_list, 2, days_list)
	print 'Number of Clusters based on Datetime = ',
	print len(ideal_cluster_list)
	return date_label_pred


def generate_location_label_pred(tweets_dict):
	"""Not being used as of now"""
	location_list = get_location_list(tweets_dict, 3, 4)
	cluster_list_locations = []
	location_label_pred = clustering_algo(cluster_list_locations, 3, location_list, 0.01)
	print location_label_pred
	print len(cluster_list_locations)
	return location_label_pred


def binary_voting(sample_num, feature_num, weights, label_pred_list):
	print '\n\n**************************Getting into final data*******************************\n'
	final_cluster_list = []
	final_label_pred_list = []
	final_label_pred_list = super_clustering_algo(final_cluster_list, sample_num, feature_num, weights,
													 label_pred_list)
	print '\nFinal Number of Clusters :',
	print len(final_cluster_list)
	return [final_cluster_list, final_label_pred_list]

def print_clusters(cluster_list):
	tf_idf_param = 0.08
	max_words_to_print = 10
	d = {}
	cnum = 0
	for fcl in cluster_list:
		print '\nCLUSTER NUMBER : ',
		print cnum
		for idx in xrange(0, len(feature_names)):
			d[feature_names[idx]] = fcl.centroid[idx]
		sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse = True)
		idx = 0
		for item in sorted_d:
			if idx >= max_words_to_print:
				break
			elif item[1] > tf_idf_param:
				print item[0],
			else:
				break
			idx += 1
		d.clear()
		sorted_d[:] = []
		cnum += 1

if __name__ == "__main__":
	'''
	Dictionary containing sample_num lists of metadata:
	Tweet Text, Upload_Date (Datetime.date)
	Keys : Serial Numbers 0 to sample_num
	'''
	# Reading data dictionary from pre-made pickle file
	# dict keys : 0 to sample_num - 1 (inclusive)
	# dict values : list -> [title(str), description(str), upload_date(date), latitude(float), longitude(float)] 
	tweets_dict = pickle.load( open( "tweets_dict.p", "rb" ) )
	sample_num = len(tweets_dict)
	print 'Number of Tweet Samples : ',
	print sample_num
	# Number of features = 2
	feature_num = 2
	# For label_pred_list, we have that:
	# [0] : title_label_pred / text_label_pred
	# [1] : date_label_pred
	label_pred_list= generate_all_label_pred(tweets_dict)
	# Weight parameters for text and date for binary voting
	weights = [0.70, 0.30]

	final_cluster_list, final_label_pred_list = binary_voting(sample_num, feature_num, weights, label_pred_list)
	
	print_choice =  raw_input('Print word summary for each cluster? Y/N : ')
	while print_choice not in ['Y','N']:
		print_choice =  raw_input('Please enter valid choice Y/N : ')
	if print_choice == 'Y':
		print_clusters(final_cluster_list)

	# pickle.dump(feature_names, open( "feature_names_list.p", "wb" ) )	
	# pickle.dump(final_cluster_list, open( "final_cluster_list.p", "wb" ) )



