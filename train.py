import numpy as np
import math
from collections import defaultdict
from predict import *

class Node:
	"""
	This is the main node class of the decision tree
	This class contains the skeleton of the structure 
	of each node in the decision tree. 

	"""
	def __init__(self, predicted_class, depth):
		'''
		This function is the constructor to initialize the 
		values in the node object.
		Parameters
		----------
		predecited_class : The predicted class for a particular node
		depth : Depth at which the node is present in the decision tree
		left : left child of the node in the decision tree
		right : Right child of the node in the decision tree
		feature_index : The index of the feature this node will divide the data into
		threshold : The threshold for the feature by which the dataset is further divided into childern of this node

		'''
		self.predicted_class = predicted_class
		self.depth = depth
		self.left = None
		self.right = None
		self.feature_index = 0
		self.threshold = 0



def import_data(file_name):

	'''
	Imports the data from the given ‘heart.dat’ file and converts it 
	to a numpy array with the elements being in float data type.

	Parameter
	---------
	file_name:  the name opf the file which contains the data to be read
	
	Returns
	-------
	train_XY :  a ndarray after reading the data from file.
	'''
	train_XY = np.genfromtxt(file_name, delimiter=' ', dtype=np.float64)
	# print(type(train_XY))
	return train_XY

def get_validation_and_train_data(train_XY_data, seed):

	'''
	This function randomly shuffles the data imported from the function 
	above and does a 80:20 split to get the training and testing data. 
	The data is further divided into train_X, train_Y, test_X, test_Y sets 
	which denote respectively the training attributes, target values for 
	training, testing attributes and target values for testing.
	
	Parameters
	----------
	train_XY_data:  the dataset provided'
	seed :          seed for random shuffluing of data before spliting.  
	
	Returns
	-------
	train_X:        Contains the 80% data points collected from the 
					randomly shuffled dataset which will be used 
					for training. length=0.8*n
	train_Y:        Provided output for the training data points
	test_X:         Contains the 20% data points collected from the 
					randomly shuffled dataset which will be used 
					for testing. length=0.2*n
	test_Y:         Provided output for the training data points
	'''

	np.random.seed(seed)
	np.random.shuffle(train_XY_data)
	size = len(train_XY_data)
	train_XY = train_XY_data[:int(0.8*size)]
	test_XY = train_XY_data[int(0.8*size):]
	train_X = train_XY[: , :-1]
	train_Y = train_XY[:, -1]
	test_X = test_XY[: , :-1]
	test_Y = test_XY[:, -1]
	return train_X, train_Y, test_X, test_Y

def get_entropy(Y):
	'''
	This function takes input a list of values of a particular feature
	and calculates entropy based on the list.

	Parameters
	----------
	Y:      The list of values provided

	Returns
	-------
	entropy: The entropy calculated for the given list
	'''
	# takes out all the unique values
	classes = list(set(Y))
	# takes lenght of the input provided
	m = len(Y)
	entropy = 0
	for i in range(len(classes)):
		# counts the occurence of a particular value
		cnt = np.sum(Y == classes[i])
		# calculates entropy for this particluar value using formula -(p(i) * log2 p(i))
		# adds it to the final answer
		entropy += -(cnt/m)*math.log2(cnt/m)
	return entropy

def calculate_gini_index(Y_subsets):
	'''
	This function is used to calculate the gini index to choose the
	best attribute that should be used for splitting at a particular
	node. Lower the gini index for an attribute the better it is.

	Parameters
	----------
	Y:      The list of values provided

	Returns
	-------
	gini_index: The gini index for the data provided
	'''
	total = 0
	gini_index = 0
	l = []
	for i in range(len(Y_subsets)):
		a = Y_subsets[i]
		m = len(a)
		gini_impurity = 1
		d = defaultdict(lambda: 0)
		for j in range(len(a)):
			total += 1
			d[a[j]] += 1
		for key in d:
			p = d[key] / m
			gini_impurity -= p ** 2
		b = [gini_impurity, m]
		l.append(b)
	for i in range(len(l)):
		gini_index += (l[i][1] / total) * l[i][0]
	return gini_index

def calculate_information_gain(Y, Y_subsets):
	'''
	This function is used to calculate the information gain to choose
	the best attribute that should be used for splitting at a particular
	node. Higher the information gain for an attribute the better it is.

	Parameters
	----------
	Y:          The list of values provided
	Y_subsets:  

	Returns
	-------
	information_gain: The information gain for the data provided
	'''
	information_gain = get_entropy(Y)
	for i in range(len(Y_subsets)):
		a = Y_subsets[i]
		information_gain -= (get_entropy(a)*len(a))/len(Y)
	return information_gain

def split_data(X, Y, feature_index, threshold):
	'''
	This function is used to split the data present at a node into the 
	left subpart and right subpart based on the attribute and threshold 
	chosen at this node using gini index or information gain.

	Parameters
	----------
	X:              list of Attributes
	Y:              Target Values
	feature_index:  Feature chosen at this node on basis of which data will be splitted
	threshold:      threshold chosen for this feature on basis of which data will be splitted

	Returns
	-------
	left_X:         attributres for the left subpart        
	left_Y:         target values for the left subpart
	right_X:        attributes for the right subpart
	right_Y:        target values for the right subpart
	'''
	left_X = np.zeros((1, X.shape[1]))
	left_Y = []
	right_X = np.zeros((1, X.shape[1]))
	right_Y = []
	for i in range(len(X)):
		a = X[i]
		if a[feature_index] < threshold:
			a = a.reshape(1, len(a))
			left_Y.append(Y[i])
			left_X = np.vstack((left_X, a))
		else:
			a = a.reshape(1, len(a))
			right_Y.append(Y[i])
			right_X = np.vstack((right_X, a))
	return left_X[1:], left_Y, right_X[1:], right_Y

def get_best_split_gini_index(X, Y):
	'''
	This function Chooses which feature is to be used to divide the
	tree by calculating gini index and taking the feature with lowest
	gini index.

	Parameters
	----------
	X:      Contains the data samples on the basis of which the 
			decision tree is to be constructed.
	Y:      The predicted values of the data samples
  
	Returns
	-------
	best_feature : the feature which gives the lowest gini index
	best_threshold : the threshold on which the best_feature gives 
	the lowest gini index
	'''
	l = []
	# traverse for every coloumn in the 2D array
	for j in range(X.shape[1]):
		# takes out a column
		a = X[:, j]
		'''
		traverse for every value in that column each value is 
		considered once as threshold and then whichever one gives 
		the best result is finally taken.
		'''
		for i in range(len(a)):
			# split the data by providing the feature index and the threshold
			left_X, left_Y, right_X, right_Y = split_data(X, Y, j, a[i])
			# ignore if the partition gives left part or right part 0
			if len(left_X) == 0 or len(right_X) == 0:
				continue
			# otherwise calclute the gini index and store that in an array
			# along with the feature index and threshold
			else:
				gini_index = calculate_gini_index([left_Y, right_Y])
				b = [gini_index, j, a[i]]
				l.append(b)
	# sort the array created in ascending order
	l.sort(key=lambda x: (x[0], x[1], x[2]))
	# take the best_feature and best_threshold
	best_feature = l[0][1]
	best_threshold = l[0][2]
	return best_feature, best_threshold

def get_best_split_information_gain(X, Y):
	'''
	This function Chooses which feature is to be used to divide the
	tree by calculating gini index and taking the feature with highest
	information gain.

	Parameters
	----------
	X:      Contains the data samples on the basis of which the 
			decision tree is to be constructed.
	Y:      The predicted values of the data samples
  
	Returns
	-------
	best_feature : the feature which gives the highest info. gain
	best_threshold : the threshold on which the best_feature gives 
	the highest info. gain.
	'''
	l = []
	# traverse for every coloumn in the 2D array
	for j in range(X.shape[1]):
		# takes out a column
		a = X[:, j]
		'''
		traverse for every value in that column each value is 
		considered once as threshold and then whichever one gives 
		the best result is finally taken.
		'''
		for i in range(len(a)):
			# split the data by providing the feature index and the threshold
			left_X, left_Y, right_X, right_Y = split_data(X, Y, j, a[i])
			# ignore if the partition gives left part or right part 0
			if len(left_X) == 0 or len(right_X) == 0:
				continue
			# otherwise calclute the info gain and store that in an array
			# along with the feature index and threshold
			else:
				information_gain = calculate_information_gain(Y, [left_Y, right_Y])
				b = [information_gain, j, a[i]]
				l.append(b)
	# sort the array created in ascending order with neagative info gain
	# that will give descending order of the array
	l.sort(key=lambda x: (-x[0], x[1], x[2]))
	# take the best_feature and best_threshold
	best_feature = l[0][1]
	best_threshold = l[0][2]
	return best_feature, best_threshold

def construct_tree_using_gini_index(X, Y, min_size, max_depth, depth):
	'''
	This function is the main function the handles the construction 
	of the entire decision tree handling all the steps. This function 
	implements the ID3 algorithm with the help of gini index to divide 
	the data at each node.

	Parameters
	----------
	X:      Contains the data samples on the basis of which the 
			decision tree is to be constructed.
	Y:      The predicted values of the data samples
	depth:  this represents the current depth upto which the 
			tree has already been built.
	min_size: minimum size of the decision tree to be built.
	max_depth: maximum depth of the decision tree to be built.

	Returns
	-------
	node: the head of he decision tree rooted in the current root
	'''
	classes = list(set(Y))
	predicted_class = classes[np.argmax([np.sum(Y == c) for c in classes])]
	node = Node(predicted_class, depth)
	if len(set(Y)) == 1:
		return node
	if depth >= max_depth:
		return node
	if len(Y) <= min_size:
		return node
	feature_index, threshold = get_best_split_gini_index(X, Y)
	node.feature_index = feature_index
	node.threshold = threshold
	left_X, left_Y, right_X, right_Y = split_data(X, Y, feature_index, threshold)
	node.left = construct_tree_using_gini_index(left_X, left_Y, min_size, max_depth, depth + 1)
	node.right = construct_tree_using_gini_index(right_X, right_Y, min_size, max_depth, depth + 1)
	return node

def construct_tree_using_information_gain(X, Y, min_size, max_depth, depth):
	'''
	This function is the main function the handles the construction 
	of the entire decision tree handling all the steps. This function 
	implements the ID3 algorithm with the help of information gain to 
	divide the data at each node.

	Parameters
	----------
	X:      Contains the data samples on the basis of which the 
			decision tree is to be constructed.
	Y:      The predicted values of the data samples
	depth:  this represents the current depth upto which the 
			tree has already been built.
	min_size: minimum size of the decision tree to be built.
	max_depth: maximum depth of the decision tree to be built.

	Returns
	-------
	node: the head of he decision tree rooted in the current root
	'''
	classes = list(set(Y))
	predicted_class = classes[np.argmax([np.sum(Y == c) for c in classes])]
	node = Node(predicted_class, depth)
	if len(set(Y)) == 1:
		return node
	if depth >= max_depth:
		return node
	if len(Y) <= min_size:
		return node
	feature_index, threshold = get_best_split_information_gain(X, Y)
	node.feature_index = feature_index
	node.threshold = threshold
	left_X, left_Y, right_X, right_Y = split_data(X, Y, feature_index, threshold)
	node.left = construct_tree_using_information_gain(left_X, left_Y, min_size, max_depth, depth + 1)
	node.right = construct_tree_using_information_gain(right_X, right_Y, min_size, max_depth, depth + 1)
	return node


def get_best_root(train_XY):
	'''
	This function takes the training dataset and find the tree with best 
	accuracy by doing 10 random splits of the data and constructs tree
	by using both the impurity measures gini index and information gain
	and take the tree with the best accuracy, also this function computes
	the average accuracy of the tree onstructed using gini index and info
	gain.

	Parameters
	----------
	train_XY:	the training dataset on which the trees are to be generated 
   
	Returns
	-------
	best_root: 	the root of the best tree contructed in terms of accuracy
	average_accuracy_gini_index:		the average accuarcy of 10 trees constructed
										using gini index as impurity measure.
	average_accuracy_information_gain:	the average accuarcy of 10 trees constructed
										using info gain as impurity measure.
	'''
	# initializing the best root to none
	best_root = None
	# initializing the best accuracy to be 0
	best_accuracy = 0
	# initializing the average accuracy using gini index to be 0
	average_accuracy_gini_index = 0
	# initializing the average accuracy using info gain to be 0
	average_accuracy_information_gain = 0
	# loop for random splitting data 10 times
	for i in range(10):
		print("Current iteration of best root : " , i+1)
		# splitting of the dataset using unique seed everytime 
		train_X, train_Y, test_X, test_Y = get_validation_and_train_data(train_XY, 10*i + i**2 + i**3)
		# tree construction using gini index
		root_using_gini_index = construct_tree_using_gini_index(train_X, train_Y, 0, 100, 0)
		# tree using information gain
		root_using_information_gain = construct_tree_using_information_gain(train_X, train_Y, 0, 100, 0)
		# computing the average accuracies of trees using gini index
		accuracy_using_gini_index = check_accuracy(root_using_gini_index, test_X, test_Y)
		# computing the average accuracies of trees using info gain
		accuracy_using_information_gain = check_accuracy(root_using_information_gain, test_X, test_Y)
		average_accuracy_gini_index += accuracy_using_gini_index
		average_accuracy_information_gain += accuracy_using_information_gain
		# comparing the accuracies of trees generated with the best accuracy
		if accuracy_using_gini_index > best_accuracy:
			best_accuracy = accuracy_using_gini_index
			best_root = root_using_gini_index
		if accuracy_using_information_gain > best_accuracy:
			best_accuracy = accuracy_using_information_gain
			best_root = root_using_information_gain
	return best_root, average_accuracy_gini_index/10, average_accuracy_information_gain/10



def prune(original_root, root, test_X, test_Y):
	'''
	This function is used for pruning the tree using Reduced Error Pruning method.

	Parameters
	----------
	original_root:  root of the tree to be pruned, this is kept as it is to compare
					accuracy of the tree new created
	root:           root of the tree to be pruned, final pruned tree is stored in this
	test_X:         test set with the help of which we prune the tree.
	test_Y:         target values of the test set provided. 
	
	Returns
	-------
	root:           final tree after pruning operations.
	'''
	if root.left == None and root.right == None:
		return root
	if root.left != None:
		root.left = prune(original_root, root.left, test_X, test_Y)
	if root.right != None:
		root.right = prune(original_root, root.right, test_X, test_Y)

	original_accuracy = check_accuracy(original_root, test_X, test_Y)
	original_left = root.left
	original_right = root.right
	root.left = None
	root.right = None

	new_accuracy  = check_accuracy(original_root, test_X, test_Y)

	if new_accuracy < original_accuracy:
		root.left = original_left
		root.right = original_right

	return root
