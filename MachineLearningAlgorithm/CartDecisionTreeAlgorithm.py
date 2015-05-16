from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix,roc_curve
import StringIO
from sklearn import svm
import time

import numpy as np
class CartDecisionTreeAlgorithm:
    def __init__(self,train_file,test_file):
        self.train_file = train_file
        self.test_file = test_file

    def classify_random_forest(self,X,Y):
       rf = RandomForestClassifier(n_estimators=400,max_features='log2')
       return rf.fit(X,Y)

    def classify(self,X,Y):
        return tree.DecisionTreeClassifier().fit(X,Y)

    def classify_SVM(self,X,Y):
        svmmodel = svm.SVC()
       # return svmmodel.fit(X,Y)
        return svmmodel

    def loadData(self,file_name):
        with open(file_name) as f:
            data = []
            for line in f:
                line = line.strip().split(",")
                data.append([x for x in line])

        return  data

    def learnSVM(self):
        train_input_data = self.loadData(self.train_file)
        target = [x[1] for x in train_input_data]
        target = target[1:]
        features = [x[2:] for x in train_input_data]
        features = features[1:]
        model = self.classify_SVM(features,target)

        test_input_data = self.loadData(self.test_file)
        actualOutput = [x[1] for x in test_input_data]
        actualOutput = actualOutput[1:]
        features = [x[2:] for x in test_input_data]
        features = features[1:]

        predictedOutput = model.predict(features)
        #print predicte dOutput
        #print actualOutput
        self.computeAccuracy(predictedOutput,actualOutput)
        print "Precision recall F score support metrics for SVM "
        print precision_recall_fscore_support(actualOutput,predictedOutput)
        print "confusion matrix"
        print confusion_matrix(actualOutput,predictedOutput)





    def learnRF(self):
        train_input_data = self.loadData(self.train_file)
        target = [x[1] for x in train_input_data]
        target = target[1:]
        features = [x[2:] for x in train_input_data]
        features = features[1:]
        model = self.classify_random_forest(features,target)

        test_input_data = self.loadData(self.test_file)
        actualOutput = [x[1] for x in test_input_data]
        actualOutput = actualOutput[1:]
        features = [x[2:] for x in test_input_data]
        features = features[1:]

        predictedOutput = model.predict(features)
        #print predictedOutput
        #print actualOutput
        self.computeAccuracy(predictedOutput,actualOutput)
        print "Precision recall F score support metrics for CART "
        print precision_recall_fscore_support(actualOutput,predictedOutput)
        print "confusion matrix"
        print confusion_matrix(actualOutput,predictedOutput)


    def learnCART(self):
        train_input_data = self.loadData(self.train_file)
        target = [x[1] for x in train_input_data]
        target = target[1:]
        features = [x[2:] for x in train_input_data]
        features = features[1:]
        model = self.classify(features,target)

        test_input_data = self.loadData(self.test_file)
        actualOutput = [x[1] for x in test_input_data]
        actualOutput = actualOutput[1:]
        features = [x[2:] for x in test_input_data]
        features = features[1:]

        predictedOutput = model.predict(features)
        #print predictedOutput
        #print actualOutput
        self.computeAccuracy(predictedOutput,actualOutput)
        print "Precision recall Fscore support metrics for CART "
        print precision_recall_fscore_support(actualOutput,predictedOutput)
        print "\nconfusion matrix\n"
        print confusion_matrix(actualOutput,predictedOutput)
        self.printDTRules(model)


    def printDTRules(self,model):
        dot_data = StringIO.StringIO()
        #with open("rules_1L.dot","w") as output_file:
        out = tree.export_graphviz(model, out_file="rules_1L.dot")




    def computeAccuracy(self,predictedOutput,actualOutput):
        count = 0
        for i in range(len(predictedOutput)):
            if predictedOutput[i] == actualOutput[i]:
                count = count +1
        print "Accuracy for model is "
        print float(count)/float(len(predictedOutput))


print "Decision Tree\n"
start_time = time.time()
obj = CartDecisionTreeAlgorithm('../clean_data/clean_train_1L.csv','../clean_data/clean_test_1k.csv')
obj.learnCART()
time_elapsed = time.time() - start_time
print "Time taken " + str(time_elapsed)

#obj.learnSVM()