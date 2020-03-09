from sklearn import datasets,neighbors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

iris = datasets.load_iris()
# 19007740
kingsID = [1,9,0,0,7,7,4,0]
Stest = np.array([[kingsID[0],kingsID[1],kingsID[2],kingsID[3],kingsID[4],kingsID[5],kingsID[6]],
[kingsID[1],kingsID[2],kingsID[3],kingsID[4],kingsID[5],kingsID[6],kingsID[0]],
[kingsID[2],kingsID[3],kingsID[4],kingsID[5],kingsID[6],kingsID[0],kingsID[1]],
[kingsID[3],kingsID[4],kingsID[5],kingsID[6],kingsID[0],kingsID[1],kingsID[2]]])

Stest = Stest/np.array([2.3,4,1.5,4]).reshape(-1,1)
Xtest = Stest+np.array([4,2,1,0]).reshape(-1,1)
Xtest = np.transpose(Xtest)
y = iris.target

neighbor1 = neighbors.KNeighborsClassifier(3, weights='distance')
neighbor1.fit(iris.data, iris.target)
print(neighbor1.predict(Xtest))

neighbor2 = neighbors.KNeighborsClassifier(7, weights='distance')
neighbor2.fit(iris.data, iris.target)
print(neighbor2.predict(Xtest))