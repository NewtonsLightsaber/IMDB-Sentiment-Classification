from preProcessing import featureProcessing
from sklearn.linear_model import LogisticRegression
import numpy

processedData = featureProcessing()
X_train = processedData[0]
X_test = processedData[1]
Y_train = processedData[2]

logReg = LogisticRegression()

#Fitting onto training data
logReg.fit(X_train,Y_train)

#predict on all of test
predictions = logReg.predict(X_test)

numpy.savetxt("logRegOutput.csv", predictions, delimiter=',', header="Output", comments="")


from sklearn.tree import DecisionTreeClassifier


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                                     max_depth=10, min_samples_leaf=5)

clf_entropy.fit(X_train, Y_train)

predictions = clf_entropy.predict(X_test)

numpy.savetxt("decisionTreeOutput.csv", predictions, delimiter=',', header="Output", comments="")
