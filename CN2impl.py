import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
import collections as clc

class CN2algorithm():

	def __init__(self,train_data_csv,test_data_csv):
		"""
		constructor: partitions data into train and test sets, sets the minimum accepted significance value
		and maximum star size which limits the number of complexes considered for specialisation.
		"""
		# train_data = pd.read_csv(train_data_csv)
		# test_data = pd.read_csv(test_data_csv)		

		# class_column_name = unsplit_data.columns[-1]
		# unsplit_data.rename(columns={class_column_name: 'class'}, inplace = True)
		# self.unsplit_data = unsplit_data
		#train_set,test_set = train_test_split(unsplit_data,test_size=0.33, random_state=42)
		self.train_set = pd.read_csv(train_data_csv)
		self.test_set =  pd.read_csv(test_data_csv)
		#self.train_set = pd.read_csv(data_csv)
		self.min_significance = 0.7
		self.max_star_size = 5

	def fit_CN2(self):
		"""
		Function to fit the CN2 model to 
		"""

		#import ipdb;ipdb.set_trace(context=8)
				
		selectors = self.find_attribute_pairs()
		remaining_examples = self.train_set
		rule_list = []
		# loop until data is all covered.
		while len(remaining_examples) >= 1:
			best_new_rule_significance = 1
			rules_to_specialise = []
			existing_results = pd.DataFrame()
			#search rule space until rule best_new_rule_significance = 1significance is lower than user set boundary(0.5 for testing)
			while  best_new_rule_significance > 0.5:
				#calls statement if its first iteration of loop
				if len(rules_to_specialise) == 0:
					ordered_rule_results = self.apply_and_order_rules_by_score(selectors, remaining_examples)
					trimmed_rule_results = ordered_rule_results[0:self.max_star_size]
				elif len(rules_to_specialise) != 0: 		 
					specialised_rules = self.specialise_complex(rules_to_specialise,selectors)
					#import ipdb;ipdb.set_trace(context=8)
					ordered_rule_results = self.apply_and_order_rules_by_score(specialised_rules, remaining_examples)
					trimmed_rule_results = ordered_rule_results[0:self.max_star_size]
				#append newly discovered rules to existing ones, order them and then take best X(3 for testing)
				existing_results = existing_results.append(trimmed_rule_results)
				existing_results =  self.order_rules(existing_results).iloc[0:2]
				#update 'rules to specialise' and significance value of best new rule 
				rules_to_specialise = trimmed_rule_results['rule']
				best_new_rule_significance = trimmed_rule_results['significance'].values[0]
				#ipdb.set_trace(context=8)
			
			#import ipdb;ipdb.set_trace(context=8)
			best_rule = (existing_results['rule'].iloc[0],existing_results['predict_class'].iloc[0],
							existing_results['num_insts_covered'].iloc[0])
			best_rule_coverage_index, best_rule_coverage_df = self.complex_coverage(best_rule[0], remaining_examples)
			rule_list.append(best_rule)
			remaining_examples = remaining_examples.drop(best_rule_coverage_index)

		return rule_list 

	def test_fitted_model(self, rule_list, data_set = 'default'):
		"""
		Test rule list returned by fit_CN2 function on test data(or manually supplied data)
		returns a dataframe that contains the rule, rule acc, num of examples covered.
		Also return general accuracy as average of each rule accuracy
		"""
		if type(data_set) == str:
			data_set = self.test_set

		remaining_examples = data_set
		list_of_row_dicts = []

		for rule in rule_list:
			rule_coverage_indexes,rule_coverage_dataframe = self.complex_coverage(rule[0],remaining_examples)
			#check for zero coverage due to noise(lense data too small)
			if len(rule_coverage_dataframe) == 0:
				row_dictionary = {'rule':rule,'pred_class':'zero coverage','rule_acc':0,
							   'num_examples':0,'num_correct':0,
							   'num_wrong':0}
				list_of_row_dicts.append(row_dictionary)			   
			#otherwise generate statistics about rule then save and remove examples from the data and test next rule.
			else:				   
				class_of_covered_examples = rule_coverage_dataframe['class']
				#import ipdb;ipdb.set_trace(context=8)
				class_counts = class_of_covered_examples.value_counts()
				rule_accuracy = class_counts.values[0]/sum(class_counts)
				num_correctly_classified_examples = class_counts.values[0]
				num_incorrectly_classified_examples = sum(class_counts.values) - num_correctly_classified_examples

				row_dictionary = {'rule':rule,'pred_class':rule[1],'rule_acc':rule_accuracy,
								   'num_examples':len(rule_coverage_indexes),'num_correct':num_correctly_classified_examples,
								   'num_wrong':num_incorrectly_classified_examples}
				list_of_row_dicts.append(row_dictionary) 

				remaining_examples = remaining_examples.drop(rule_coverage_indexes)

		results = pd.DataFrame(list_of_row_dicts)
		overall_accuracy = sum(results['rule_acc'])/len([r for r in results['rule_acc'] if r !=0])
		return results, overall_accuracy


	def apply_and_order_rules_by_score(self, complexes, data_set = 'default'):
		"""
		A function which takes a list of complexes/rules and returns a pandas DataFrame
		that contains the complex, the entropy, the significance, the number of selectors,
		the number of examples covered, the length of the rule and the predicted class of the rule. 
		The input param complexes should be a list of lists of tuples.
		"""
		#import ipdb;ipdb.set_trace(context=8)
		
		#build a dictionary for each rule with relevant stats
		if type(data_set) == str: 
			data_set = self.train_set
		list_of_row_dicts = []
		for row in complexes:
			rule_coverage = self.complex_coverage(row, data_set)[1]
			rule_length = len(row)
			#test if rule covers 0 examples
			if len(rule_coverage) == 0:

				row_dictionary = {'rule': row, 'predict_class':'dud rule',
								'entropy': 10, 'laplace_accuracy': 0,
								'significance':0,'length':rule_length, 
								'num_insts_covered':0,'specificity':0}
				list_of_row_dicts.append(row_dictionary)
			#calculate stats for non 0 coverage rules
			else:

				num_examples_covered = len(rule_coverage)
				entropy_of_rule = self.rule_entropy(rule_coverage)
				significance_of_rule = self.rule_significance(rule_coverage)
				laplace_accuracy_of_rule = self.rule_laplace_accuracy(rule_coverage)
				class_attrib = rule_coverage['class']
				#import ipdb;ipdb.set_trace(context=8)
				class_counts = class_attrib.value_counts()
				majority_class = class_counts.axes[0][0]
				rule_specificity = class_counts.values[0]/sum(class_counts) 
				row_dictionary = {'rule': row, 'predict_class':majority_class,
									'entropy': entropy_of_rule, 'laplace_accuracy': laplace_accuracy_of_rule,
									'significance':significance_of_rule,'length':rule_length, 
									'num_insts_covered':num_examples_covered,'specificity':rule_specificity}
				list_of_row_dicts.append(row_dictionary)
		#put dictionaries into dataframe and order them according to laplace acc, length
		rules_and_stats = pd.DataFrame(list_of_row_dicts)
		ordered_rules_and_stats = self.order_rules(rules_and_stats)	

		return ordered_rules_and_stats

	
	def order_rules(self, dataFrame_of_rules):
		"""
		Function to order a dataframe of rules and stats according to laplace acc and length then reindex
		the ordered frame.
		"""
		ordered_rules_and_stats = dataFrame_of_rules.sort_values(['entropy', 'length',
			                                                       'num_insts_covered'], ascending=[True, True, False])
		ordered_rules_and_stats = ordered_rules_and_stats.reset_index(drop=True)	

		return ordered_rules_and_stats	


	def find_attribute_pairs(self):
		"""function to return the first set 
		   of complexes which are the
	       1 attribute selectors
		"""


		#get attribute names
		attributes = self.train_set.columns.values.tolist()
		
		#remove class from list of attributes
		del attributes[-1]
		
		#get possible values for attributes 
		possAttribVals = {} 	
		for att in attributes:
			possAttribVals[att] = set(self.train_set[att])

		# get list of attribute,value pairs
		# from possAttribVals dictionary
		attrib_value_pairs = []
		for key in possAttribVals.keys():
			for possVal in possAttribVals[key]:
				attrib_value_pairs.append([(key,possVal)])

		return attrib_value_pairs

	def specialise_complex(self, target_complexes, selectors):
		"""
		Function to specialise the complexes in the "star", the current set of
		complexes in consideration. Exepects to receive a complex (a list of tuples)
		to which it adds addtional conjunctions using all the possible selectors. Returns
		a list of new, specialised complexes.
		"""
		#import ipdb;ipdb.set_trace(context=8)

		provisional_specialisations = []
		for targ_complex in target_complexes: 
			for selector in selectors:
				#check to see if target complex is a single tuple otherwise assume list of tuples
				if type(targ_complex) == tuple:
					comp_to_specialise = [copy.copy(targ_complex)]
				else:
					comp_to_specialise = copy.copy(targ_complex)	
				
				comp_to_specialise.append(selector[0])
				
				#count if any slector is duplicated and append rule if not
				count_of_selectors_in_complex = clc.Counter(comp_to_specialise)
				flag = True
				for count in count_of_selectors_in_complex.values():
					if count > 1:
						flag = False


				if flag == True:
					provisional_specialisations.append(comp_to_specialise)
		

		#remove complexes that have been specialised with same selector eg [(A=1),(A=1)]
		#trimmed_specialisations = [rule for rule in provisional_specialisations if rule[0] != rule[1]]
	
		return provisional_specialisations		


	def build_rule(self,passed_complex):
		"""
		build a rule in dict format where target attributes have a single value and non-target attributes
		have a list of all possible values. Checks if there are repetitions in the attributes used, if so
		it returns False
		"""
		atts_used_in_rule = []
		for selector in passed_complex:
			atts_used_in_rule.append(selector[0])
		set_of_atts_used_in_rule = set(atts_used_in_rule)
		 
		if len(set_of_atts_used_in_rule) < len(atts_used_in_rule):
			return False


		rule = {}
		attributes = self.train_set.columns.values.tolist() 	
		for att in attributes:
			rule[att] = list(set(self.train_set[att]))

		for att_val_pair in passed_complex:
			att = att_val_pair[0]
			val = att_val_pair[1]
			rule[att] = [val]
		return rule		


	def complex_coverage(self, passed_complex, data_set = 'default'):
		""" Returns set of instances of the data 
		    which complex(rule) covers as a dataframe.
		"""
		if type(data_set) == str: 
			data_set = self.train_set
		coverage = []

		rule = self.build_rule(passed_complex)
		if rule == False:
			return [],[]

		mask = data_set.isin(rule).all(axis=1)
		rule_coverage_indexes = data_set[mask].index.values
		rule_coverage_dataframe = data_set[mask]
       
		# #iterate over dataframe rows
		# for index,row in data_set.iterrows():
		# 	if self.check_rule_datapoint(row, complex):
		# 		coverage.append(index)
		

		#import ipdb;ipdb.set_trace(context=8)
		return rule_coverage_indexes, rule_coverage_dataframe

	def check_rule_datapoint(self, datapoint, complex):
		"""
		Function to check if a given data point satisfies
        the conditions of a given complex. Data point 
        should be a pandas Series. Complex should be a
        tuple or a list of tuples where each tuple is of
        the form ('Attribute', 'Value').
        """
		if type(complex) == tuple:
			if datapoint[complex[0]] == complex[1]:
				return True
			else:
				return False
		#import ipdb;ipdb.set_trace(context=8)
		if type(complex) == list:
			result = True	
			for selector in complex:
				if datapoint[selector[0]] != selector[1]:
					result = False

			return result

	def rule_entropy(self,covered_data):
		"""
		Function to check the Shannon entropy of a complex/rule
		given the instances it covers. Pass the instances 
		covered by the rule as a dataframe where class cloumn is
		named class.
		""" 
		class_series = covered_data['class']
		num_instances = len(class_series)
		class_counts = class_series.value_counts()
		class_probabilities = class_counts.divide(num_instances)
		log2_of_classprobs = np.log2(class_probabilities)
		plog2p = class_probabilities.multiply(log2_of_classprobs)
		entropy = plog2p.sum()*-1
		
		return entropy

	def rule_significance(self,covered_data):
		"""
		Fucntion to check the significance of a rule using the 
		likelihood ratio test where observed frequency of class
		in the coverage of the rule is compared to the observed
		frequencies of the classes in the training data.
		"""
		covered_classes = covered_data['class']
		covered_num_instances = len(covered_classes)
		covered_counts = covered_classes.value_counts()
		covered_probs = covered_counts.divide(covered_num_instances)

		train_classes = self.train_set['class']
		train_num_instances = len(train_classes)
		train_counts = train_classes.value_counts()
		train_probs = train_counts.divide(train_num_instances)

		significance = covered_probs.multiply(np.log(covered_probs.divide(train_probs))).sum()*2

		return significance

	def rule_laplace_accuracy(self,covered_data):
		"""
		function to calculate laplace accuracy of a rule, taken from update to CN2
		paper by author of original CN2.
		"""
		#import ipdb;ipdb.set_trace(context=8)

		class_series = covered_data['class']
		class_counts = class_series.value_counts()
		num_instances = len(class_series)
		num_classes = len(class_counts)
		num_pred_class = class_counts.iloc[0]
		#laplace_accuracy = (num_pred_class+1)/(num_instances+num_classes)
		laplace_accuracy_2 = (num_instances + num_classes - num_pred_class - 1)/(num_instances + num_classes)
		return laplace_accuracy_2

if __name__ == '__main__':
	"""
	main method to test algorithm on 4 data sets.
	"""
	lenseFit = CN2algorithm('train_set_lense.csv','test_set_lense.csv')
	lenseRules = lenseFit.fit_CN2()
	lenseTest = lenseFit.test_fitted_model(lenseRules,lenseFit.test_set)[0]
	lenseTest.to_csv('lense_test_results.csv')

	zooFit = CN2algorithm('train_set_zoo.csv','test_set_zoo.csv')
	zooRules = zooFit.fit_CN2()
	zooTest = zooFit.test_fitted_model(zooRules,zooFit.test_set)[0]
	zooTest.to_csv('zoo_test_results.csv')

	tttFit = CN2algorithm('train_set_ttt.csv','test_set_ttt.csv')
	tttRules = tttFit.fit_CN2()
	tttTest = tttFit.test_fitted_model(tttRules,tttFit.test_set)[0]
	tttTest.to_csv('ttt_test_results.csv')

	votingFit = CN2algorithm('train_set_voting.csv','test_set_voting.csv')
	votingRules = votingFit.fit_CN2()
	votingTest = votingFit.test_fitted_model(votingRules,votingFit.test_set)[0]
	votingTest.to_csv('voting_test_results.csv')



		 


	


		    
	





        






			



