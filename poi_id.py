#!/usr/bin/python

import sys
import pprint
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','exercised_stock_options', 'bonus'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
for employee in data_dict:
    print employee
for employee in data_dict:
    pprint.pprint(data_dict[employee])

### Print one example
pprint.pprint(data_dict['LAY KENNETH L'])

#What is the Travel Agency & Total?
pprint.pprint(data_dict['THE TRAVEL AGENCY IN THE PARK']) #mostly NaNs not a person
pprint.pprint(data_dict['TOTAL']) #just the Total

#Let's see if there are any duplicates
employee_names = []
for employee in data_dict:
    employee_names.append(employee)
print len(employee_names)
employee_set = set(employee_names)
print len(employee_set)  

#No duplicates...
#Maybe there are names spelled slightly different twice?
#Sort alphabetically and visually inspect
employee_names.sort()
pprint.pprint(employee_names) 
#All names except for Total and Travel Agency seem valid

###Remove the 2 outliers from the original dataset
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('TOTAL', 0)
print len(data_dict) #144 records as expected  

### Task 3: Create new feature(s)

#POIs
for employee in data_dict:
    if data_dict[employee]['poi']:
        print float(data_dict[employee]['restricted_stock'])/float(data_dict[employee]['total_stock_value'])
#Non-POIs
for employee in data_dict:
    if not data_dict[employee]['poi']:
        print float(data_dict[employee]['restricted_stock'])/float(data_dict[employee]['total_stock_value'])
        print float(data_dict[employee]['expenses'])/float(data_dict[employee]['salary'])

#About a third of POIs have all their stock as restricted_stock (1.0)
#Let's add it to the data_dict
for employee in data_dict:
    data_dict[employee]['restricted_stock_ratio'] = round(float(data_dict[employee]['restricted_stock']) / \
                                                    float(data_dict[employee]['total_stock_value']),2)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### scale features via min-max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
###FINAL CHOSEN ALGORITHM AND PARAMETERS - Better Recall, more True Positives
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_neighbors=5, p=2, weights='distance')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
test_classifier(clf, my_dataset, features_list)

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)