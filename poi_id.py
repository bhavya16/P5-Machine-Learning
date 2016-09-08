#!/usr/bin/python

import sys
import pprint
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings of features with target variable followed by predictor variables.
### The first feature should be the target variable and in our case our target variable is "poi".

features_list = ['poi','salary','exercised_stock_options', 'bonus'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dictionary = pickle.load(data_file)

### Task 2: Remove outliers
for employee in data_dictionary:
    print employee
for employee in data_dictionary:
    pprint.pprint(data_dictionary[employee])

### Print one example
pprint.pprint(data_dictionary['LAY KENNETH L'])

#What is the Travel Agency & Total?
pprint.pprint(data_dictionary['THE TRAVEL AGENCY IN THE PARK']) #mostly NaNs not a person
pprint.pprint(data_dictionary['TOTAL']) 
#Total,its not the name,it is just the total

#Checking on if there is any duplicate or not

employee_names = []    #Creating an empty list
for employee in data_dictionary:      #Iterating over to the data dictionary and appending the names of the employee in the employee_names list
    employee_names.append(employee)
print len(employee_names) #checking the length of the employee_names list
employee_set = set(employee_names)
print len(employee_set)  

#Maybe there are names spelled slightly different twice?
#Sort alphabetically and visually inspect
employee_names.sort()
pprint.pprint(employee_names) 
#All names looks valid and normal except the Total and Travel Agency where Total is not the name it is just the total and also Travel Agency is not the name of the #employee as the name suggest some Agency so we will remove these two data points since they are just misleading.

###Removing the above 2 outliers which we found are not valid
data_dictionary.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dictionary.pop('TOTAL', 0)
print len(data_dictionary) #checking the length of the dictionary after removing the outliers

#Now there are 144 data points since we removed 2 data points out of 146 data points. 

### Task 3: Creating new features

#POIs
for employee in data_dictionary:
    if data_dictionary[employee]['poi']:
        print float(data_dictionary[employee]['restricted_stock'])/float(data_dictionary[employee]['total_stock_value'])
#Non-POIs
for employee in data_dictionary:
    if not data_dictionary[employee]['poi']:
        print float(data_dictionary[employee]['restricted_stock'])/float(data_dictionary[employee]['total_stock_value'])
        print float(data_dictionary[employee]['expenses'])/float(data_dictionary[employee]['salary'])


#Adding this new feature to the data_dictionary
for employee in data_dictionary:
    data_dictionary[employee]['restricted_stock_ratio'] = round(float(data_dictionary[employee]['restricted_stock']) / \
                                                    float(data_dictionary[employee]['total_stock_value']),2)

#Added the 'restricted_stock_ratio' feature to the data dictionary by first iterating over to the data_dictionary and adding this new feature by rounding and take ratio of each employee restricted_stock to the total_stock_value

### Store to Final_dataset for easy export below.
Final_dataset = data_dictionary

### Extract features and labels from dataset for local testing
data = featureFormat(Final_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### scale features via min-max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
#1)Random Forest

#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=10) 
# Got an accuracy of 85%, Precision =0.53651  ,Recall=0.26450 and F-value of 0.29435 for Random Forest

#2)Gradient Boosting Classifier

#from sklearn.ensemble import GradientBoostingClassifier
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
#         max_depth=1, random_state=0)
# Got an accuracy of 79%, Precision =0.31110  ,Recall=0.30550 and F-value of 0.30827 for Gradient Boosting

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
###FINAL CHOSEN ALGORITHM AND PARAMETERS - Better Recall, more True Positives
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
#clf = KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
#clf = KNeighborsClassifier(n_neighbors=2, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
test_classifier(clf, Final_dataset, features_list)

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, Final_dataset, features_list)