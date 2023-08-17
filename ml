INTRODUCTION
Machine Learning :
Machine Learning is a branch of artificial intelligence(AI) and computer science which focuses
on the use of data and algorithms to imitate the way that humans learn, gradually improving its
accuracy.
Numpy :
Numpy is a python library which can be used for working with arrays, linear algebra, matrices
etc..
Numpy stands for “Numerical Python”.
The array object in numpy is called nd.array.
We can create a numpy and array object by using the array function.
Type is a built-in function in python that tells us the type of object passed to it.
Zero dimensional or 0D arrays or scalars are elements in an array.
Each value in an array is a 0D.
An array that uses a 0 dimensional array as its elements is called a unidimensional or 1D array.
An array that uses a 1 dimensional array as its elements is called a
2 dimensional array.
They are often used to represent a matrix.
Numpy arrays provide the n dim attribute, which returns an integer that tells us how many
dimensions the array has.
0 Dimensional Array :
import numpy as np
arr = np.array(42)
print(arr)
print(arr.ndim)
Output :
42 [0D]
1 Dimensional Array :
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))
print(arr.ndim)
Output :
([1, 2, 3, 4, 5])
<class 'numpy.ndarray’>
[1D]
2 Dimensional Array :
import numpy as np
arr = np.array([1, 2, 3], [4, 5, 6])
print(arr)
print(arr.ndim)
Output :
([1, 2, 3], [4, 5, 6])
[2D]
Iterating Arrays :
1. import numpy as np
arr = np.array([1, 2, 3], [4, 5, 6])
for x in arr:
print(x)
Output 1 :
[1 2 3]
[4 5 6]
2. import numpy as np
arr = np.array([1, 2, 3], [4, 5, 6])
for x in arr:
for y in x:
print(y)
Output 2 :
1
2
3
4
5
6
3. import numpy as np
arr = np.array([1, 2, 3], [4, 5, 6])
for x in np.nditer(arr):
print(x)
Output 3 :
1
2
3
4
5
6
Slicing :
Slicing in python means taking elements from one given index to another given index. We pass
the slice instead of an index like this.
If we don’t pass start. It considers zero.
If we don’t pass end. It considers the length of the array as in that dimension.
If we don’t pass step. It considers one.
Slice the elements from index 1 to index 5 from the given 1D array
arr = np.array([1, 2, 3, 4, 5])
arr[1:5]
Slice the elements from index 4 to the end of the array.
arr = np.array([1, 2, 3, 4, 5, 6, 7])
arr[4:]
Slice from the index is from the end to the index1 from the end.
[start : end] [start : end : step]
arr = np.array([1, 2, 3, 4, 5, 6, 7])
arr[-3:-1]
Shape :
Numpy arrays have an attribute called shape, that returns a tuple with each index having the
number of corresponding elements.
Print the shape of a given 2D array.
import numpy as np
arr = np.array([1, 2, 3], [4, 5, 6])
print(arr.shape)
Sort is a function used to sort a specific array. (i.e., numerical or alphabetical)
Sort any 1D array.
np.sort(arr)
1.Write a Numpy program to create a structured array from given student
name, height, class and their data types. Now sort the array on height.
Program:
import numpy as np
data_type = [('name', 'S15'), ('class', int), ('height', float)]
students_details = [('James', 5, 48.5), ('Nail', 6, 52.5),('Paul', 5, 42.10), ('Pit', 5, 40.11)]
students = np.array(students_details, dtype=data_type)
print("Original array:")
print(students)
print("Sort by height:")
print(np.sort(students, order='height'))
Output :
Original array:
[(b'James', 5, 48.5 ) (b'Nail', 6, 52.5 ) (b'Paul', 5, 42.1 ) (b'Pit', 5, 40.11)]
Sort by height:
[(b'Pit', 5, 40.11) (b'Paul', 5, 42.1 ) (b'James', 5, 48.5 ) (b'Nail', 6, 52.5 )]
2.Write a program to create an array and perform the basic operations like
sum, add, sqrt, T.
Program:
import numpy as np
a=np.array([0.4,0.5])
B = np.array([1,2])
sum=np.sum(a)
Add = np.add(a,B)
sqrt =np.sqrt([1, 4, 9, 16])
arr = np.array([[1, 2, 3], [4, 5, 6]])
transpose = gfg.T
print(transpose)
Output :
0.9
[1.4,2.5]
[1,2,3,4]
[[1 4]
[2 5]
[3 6]
3. Write a numpy program to compute determinant of square array
Program:
import numpy as np
n_array = np.array([[50, 29], [30, 44]])
det = np.linalg.det(n_array)
print("\nDeterminant of given 2X2 matrix:")
print(int(det))
Output :
Determinant of given 2X2 matrix: 1330
4.Write a numpy program to generate 6 random integers b/w 10 and 30.
Program:
import numpy as np
x = np.random.randint(low=10, high=30, size=6)
print(x)
Output :
[14 25 20 12 27 22]
5. Evaluate various classification algorithms performance on a dataset using
various measures like True Positive rate, False positive rate, precision, recall
Program :
import pandas as pd
import numpy as np
from sklearn import datasets
#Load the breast cancer dataset
bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target
from sklearn.model_selection import train_test_split
#Create training and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1,
stratify=y)
#Confusion Matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
# Standardize the data set
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#Fit the SVC model
svc = SVC(kernel = 'linear', C = 10.0, random_state = 1)
svc.fit(X_train, y_train)
#Get the predictions
y_pred = svc.predict(X_test)
#Calculate the confusion matrix
conf_matrix = confusion_matrix(y_true = y_test, y_pred = y_pred)
#Print the confusion matrix using Matplotlib
fig, ax = plt.subplots(figsize = (5, 5))
ax.matshow(conf_matrix, cmap = plt.cm.Oranges, alpha = 0.3)
for i in range (conf_matrix.shape[0]) :
for j in range (conf_matrix.shape[1]) :
ax.text(x = j, y = i, s = conf_matrix[i, j], va = 'center', ha = 'center', size = 'xx-large')
plt.xlabel('Predictions', fontsize = 18)
plt.ylabel('Actuals', fontsize = 18)
plt.title('Confusion Matrix', fontsize = 18)
plt.show()
print('Precision : %.3f' %precision_score(y_test, y_pred))
print('Recall : %.3f' %recall_score(y_test, y_pred))
print('Accuracy : %.3f' %accuracy_score(y_test, y_pred))
print('F1 : %.3f' %f1_score(y_test, y_pred))
Output :
Precision : 0.972
Recall : 0.972
Accuracy : 0.965
F1 : 0.972
6. Write a Numpy program to generate a matrix product of two arrays.
Program :
import numpy as np
x = [[1, 0], [1, 1]]
y = [[3, 1], [2, 2]]
print("Matrices and vectors.")
print("x:")
print(x)
print("y:")
print(y)
print("Matrix product of above two arrays:")
print(np.matmul(x, y))
Output :
Matrices and vectors.
x:
[[1, 0], [1, 1]]
y:
[[3, 1], [2, 2]]
Matrix product of above two arrays:
[[3 1]
[5 3]]
7. Pandas Programs :
a. Creating Series from Array
Program :
import pandas as pd
import numpy as np
info = np.array(['P','a','n','d','a','s'])
a = pd.Series(info)
print(a)
Output :
0 P
1 a
2 n
3 d
4 a
5 s
dtype: object
b. Create a Series from dict
Program :
import pandas as pd
import numpy as np
info = {'x' : 0., 'y' : 1., 'z' : 2.}
a = pd.Series(info)
print (a)
Output :
x 0.0
y 1.0
z 2.0
dtype: float64
c. Create a DataFrame from Dict of ndarrays/ Lists
Program :
import pandas as pd
info = {'ID' :[101, 102, 103],'Department' :['B.Sc','B.Tech','M.Tech',]}
df = pd.DataFrame(info)
print (df)
Output :
ID Department
0 101 B.Sc
1 102 B.Tech
2 103 M.Tech
d. Create a DataFrame from Dict of Series:
Program :
import pandas as pd
info = {'one' : pd.Series([1, 2, 3, 4, 5, 6], index=['a', 'b', 'c', 'd', 'e', 'f']),
'two' : pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
}
d1 = pd.DataFrame(info)
print (d1)
Output :
one two
a 1.0 1
b 2.0 2
c 3.0 3
d 4.0 4
e 5.0 5
f 6.0 6
g NaN 7
h NaN 8
e. Write a Pandas program to join the two given dataframes along rows and assign
all data.
Program :
import pandas as pd
student_data1 = pd.DataFrame({
'student_id': ['S1', 'S2', 'S3', 'S4', 'S5'],
'name': ['Danniella Fenton', 'Ryder Storey', 'Bryce Jensen', 'Ed Bernal', 'Kwame Morin'],
'marks': [200, 210, 190, 222, 199]})
student_data2 = pd.DataFrame({
'student_id': ['S4', 'S5', 'S6', 'S7', 'S8'],
'name': ['Scarlette Fisher', 'Carla Williamson', 'Dante Morse', 'Kaiser William', 'Madeeha
Preston'],
'marks': [201, 200, 198, 219, 201]})
print("Original DataFrames:")
print(student_data1)
print("-------------------------------------")
print(student_data2)
print("\nJoin the said two dataframes along rows:")
result_data = pd.concat([student_data1, student_data2], axis = 1)
print(result_data)
Output :
Original DataFrames:
student_id name marks
0 S1 Danniella Fenton 200
1 S2 Ryder Storey 210
2 S3 Bryce Jensen 190
3 S4 Ed Bernal 222
4 S5 Kwame Morin 199
-------------------------------------
student_id name marks
0 S4 Scarlette Fisher 201
1 S5 Carla Williamson 200
2 S6 Dante Morse 198
3 S7 Kaiser William 219
4 S8 Madeeha Preston 201
Join the said two dataframes along rows:
student_id name marks student_id name marks
0 S1 Danniella Fenton 200 S4 Scarlette Fisher 201
1 S2 Ryder Storey 210 S5 Carla Williamson 200
2 S3 Bryce Jensen 190 S6 Dante Morse 198
3 S4 Ed Bernal 222 S7 Kaiser William 219
4 S5 Kwame Morin 199 S8 Madeeha Preston 201
8. Linear Regression Program
Program :
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
def estimate_coef(x, y):
#number of observations/points
n = np.size(x)
#mean of x and y vector
m_x = np.mean(x)
m_y = np.mean(y)
#calculating cross-deviation and deviation about x
SS_xy = np.sum(y*x) - n*m_y*m_x
SS_xx = np.sum(x*x) - n*m_x*m_x
#calculating regression coefficients
b_1 = SS_xy/SS_xx
b_0 = m_y - b_1 * m_x
return(b_0, b_1)
def plot_regresssion_line(x, y, b):
#plotting the actual points as scatter plot
plt.scatter(x, y, color = "m", marker = "o", s = 30)
#predicted response vector
y_pred = b[0] + b[1]*x
#plotting the regression line
plt.plot(x, y_pred, color ="g")
#putting labels
plt.xlabel('x')
plt.ylabel('y')
#function to show plot
plt.show()
def main():
#observations/data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
#estimating coefficients
b = estimate_coef(x, y)
print("\nEstimated coefficients:\n\nb_0 = {} \nb_1 = {} \n".format(b[0], b[1]))
#plotting regression line
plot_regresssion_line(x, y, b)
if __name__=="__main__":
main()
Output :
Estimated coefficients:
b_0 = 1.2363636363636363
b_1 = 1.1696969696969697
9. Logistic Regression
Program :
import numpy as np
from sklearn import linear_model
%matplotlib inline
#Reshaped for Logistic function.
X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
logr = linear_model.LogisticRegression()
logr.fit(X, y)
#predict if tumor is cancerous where the size is 3.46mm:
predicted = logr.predict(np.array([3.47]).reshape(-1, 1))
print(predicted)
Output :
[1]
10. Program to solve linear equations using scipy
Program :
from scipy import linalg
import numpy as np
# The function takes two arrays
a = np.array([[7, 2], [4, 5]])
b = np.array([8, 10])
# Solving the linear equations
res = linalg.solve(a,b)
print(res)
Output :
[0.74074074 1.40740741]
11. Decision Tree
Program :
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
%matplotlib inline
df = pandas.read_csv("/content/drive/MyDrive/Data.csv")
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']
dtree = DecisionTreeClassifier(criterion = "entropy")
dtree = dtree.fit(X, y)
tree.plot_tree(dtree, feature_names = features)
print(dtree.predict([[40, 10, 7, 1]]))
Input : data.csv
Age,Experience,Rank,Nationality,Go
36,10,9,UK,NO
42,12,4,USA,NO
23,4,6,N,NO
52,4,4,USA,NO
43,21,8,USA,YES
44,14,5,UK,NO
66,3,7,N,YES
35,14,9,UK,YES
52,13,7,N,YES
35,5,9,N,YES
24,3,5,USA,NO
18,3,7,UK,YES
45,9,9,UK,YES
Output :
[1]
12. K-Nearest Neighbor
Program :
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import random
#Loading data
data_iris = load_iris()
#To get list of target names
label_target = data_iris.target_names
print()
print("Sample Data from Iris Dataset")
print("*"*30)
#To display the sample data from the iris dataset
for i in range(10):
rn = random.randint(0,120)
print(data_iris.data[rn], "===>", label_target[data_iris.target[rn]])
#Create feature and target arrays
X = data_iris.data
y = data_iris.target
#Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 8)
print("The Training dataset length: ", len(X_train))
print("The Testing dataset length: ", len(X_test))
try:
nn = int(input("Enter number of neighbors : "))
knn = KNeighborsClassifier(nn)
knn.fit(X_train, y_train)
#To get test data from the user
test_data = input("Enter Test Data : ").split(",")
for i in range(len(test_data)):
test_data[i] = float(test_data[i])
print()
v = knn.predict([test_data])
print("Predicted output is : ",label_target[v])
except:
print("Please supply valid input........")
Output :
Sample Data from Iris Dataset
******************************
[7.1 3. 5.9 2.1] ===> virginica
[5.6 3. 4.5 1.5] ===> versicolor
[5.7 3.8 1.7 0.3] ===> setosa
[5. 3. 1.6 0.2] ===> setosa
[6.7 3.1 4.4 1.4] ===> versicolor
[5. 3.2 1.2 0.2] ===> setosa
[4.9 3.1 1.5 0.1] ===> setosa
[4.4 3.2 1.3 0.2] ===> setosa
[5.1 3.4 1.5 0.2] ===> setosa
[5.1 3.3 1.7 0.5] ===> setosa
The Training dataset length: 120
The Testing dataset length: 30
Enter number of neighbors : 5
Enter Test Data : 7.6,3.,6.6,2.1
Predicted output is : ['virginica']
13. K-Means Clustering
Program :
from sklearn.cluster import KMeans
kmeans = KMeans(init = 'k-means++')
import numpy as np
X = np.array([[1.713,1.586], [0.180,1.786], [0.353,1.240],[0.940,1.566], [1.486,0.759],
[1.266,1.106],[1.540,0.419],[0.459,1.799],[0.773,0.186]])
y=np.array([0,1,1,0,1,0,1,1,1])
kmeans = KMeans(n_clusters=3, random_state=0).fit(X,y)
kmeans.predict([[1.713,1.586]])
Output :
array([2])
14. Naive Bayes With Python
a. Program :
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
data = {'outlook' : ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny',
'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
'temp' : ['hot','hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot',
'mild'],
'humidity' : ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal',
'normal', 'high', 'normal', 'high'],
'windy' : ['false', 'true', 'false', 'false', 'false', 'true', 'true', 'false', 'false', 'false', 'true', 'true',
'false', 'true'],
'play' : ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}
df = pd.DataFrame(data)
df
Output :
outlook tem
p
humidit
y
wind
y
pla
y
0 sunny hot high false no
1 sunny hot high true no
2
overcas
t
hot high false yes
3 rainy mild high false yes
4 rainy cool normal false yes
5 rainy cool normal true no
6
overcas
t
cool normal true yes
7 sunny mild high false no
8 sunny cool normal false yes
9 rainy mild normal false yes
10 sunny mild normal true yes
11
overcas
t
mild high true yes
12 overcas
t
hot normal false yes
13 rainy mild high true no
b. convert data type object to category
Program :
df_c = df.astype('category')
df_c["outlook"] = df_c["outlook"].cat.codes
df_c["temp"] = df_c["temp"].cat.codes
df_c["humidity"] = df_c["humidity"].cat.codes
df_c["windy"] = df_c["windy"].cat.codes
df_c["play"] = df_c["play"].cat.codes
df_c.head(14)
Output :
outloo
k
tem
p
humidit
y
wind
y
pla
y
0 2 1 0 0 0
1 2 1 0 1 0
2 0 1 0 0 1
3 1 2 0 0 1
4 1 0 1 0 1
5 1 0 1 1 0
6 0 0 1 1 1
7 2 2 0 0 0
8 2 0 1 0 1
9 1 2 1 0 1
10 2 2 1 1 1
11 0 2 0 1 1
12 0 1 1 0 1
13 1 2 0 1 0
c. splitting the label and features data
Program :
X = df_c.iloc[:, :4].values
X
Y = df_c.iloc[:, 4].values
Y
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 45)
print(X_train.shape, y_train.shape)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))
predicted = model.predict([[0, 2, 1, 1]])
print("Predicted Value : ", predicted)
Output :
(11, 4) (11,)
Accuracy : 1.0
Predicted Value : [1]
15. Fuzzy C-Means Clustering
Program :
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
np.random.seed(0)
data = np.random.rand(100,2)
n_clusters = 3
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T, n_clusters, 2, error = 0.005, maxiter =
1000, init = None)
cluster_membership = np.argmax(u, axis = 0)
print('Cluster Centers : ', cntr)
print('Cluster Membership : \n', cluster_membership)
Note :
Use - pip install scikit-fuzzy
Output :
Cluster Centers : [[0.22645397 0.71840176]
[0.52083891 0.18668653]
[0.76252289 0.60239021]]
Cluster Membership :
[2 2 0 0 2 2 2 1 0 2 2 0 0 0 1 0 0 0 2 2 1 1 2 1 1 2 1 1 1 1 1 1 0 1 1 2 2
1 1 1 1 0 1 1 2 0 0 1 1 1 1 2 0 2 0 0 1 2 2 2 2 2 0 0 1 2 1 2 2 2 2 0 2 0
2 0 0 0 2 1 2 2 2 0 1 1 1 1 0 1 0 1 2 2 1 1 0 2 1 0]
16. Make predictions using bagging for classification
Program :
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
# define dataset
X, y = make_classification(n_samples=10, n_features=5, n_informative=2, n_redundant=2,
random_state=1)
# define the model
model = BaggingClassifier()
# fit the model on the whole dataset
model.fit(X, y)
# make a single prediction
row = [[1.65980218 ,-1.04052679 , 0.89368622 , 1.03584131 ,-1.55118469]]
yhat = model.predict(row)
print('Predicted Class: %d' % yhat)
Output :
Predicted Class: 1
17. Boosting Algorithm
Program :
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
# define dataset
X, y = make_classification(n_samples=10, n_features=5, n_informative=2, n_redundant=2,
random_state=1)
cl = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
cl.fit(X, y)
print(X.shape, y.shape)
print(X)
row = [[-0.19183555, 1.05492298, -0.7290756 , -1.14651383 ,1.44634283]]
yhat = cl.predict(row)
print('Predicted Class: %d' % yhat)
predictions=[round(value) for value in yhat]
accuracy=accuracy_score(yhat,predictions)
print("Accuracy:%.2f%%" %(accuracy*100))
Output :
(10, 5) (10,)
[[-0.19183555 1.05492298 -0.7290756 -1.14651383 1.44634283]
[-1.11731035 0.79495321 3.11651775 -2.85961623 -1.52637437]
[ 0.2344157 -1.92617151 2.43027958 1.49509867 -3.42524143]
[-0.67124613 0.72558433 1.73994406 -2.00875146 -0.60483688]
[-0.0126646 0.14092825 2.41932059 -1.52320683 -1.60290743]
[ 1.6924546 0.0230103 -1.07460638 0.55132541 0.78712117]
[ 0.74204416 -1.91437196 3.84266872 0.70896364 -4.42287433]
[-0.74715829 -0.36632248 -2.17641632 1.72073855 1.23169963]
[-0.88762896 0.59936399 -1.18938753 -0.22942496 1.37496472]
[ 1.65980218 -1.04052679 0.89368622 1.03584131 -1.55118469]]
Predicted Class: 1
Accuracy:100.00%
18. Random Forest Algorithm
Program :
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn import metrics
bank_data = pd.read_csv("/content/drive/MyDrive/bank.csv")
bank_data = bank_data.loc[:, ['age', 'default', 'cons.price.idx', 'cons.conf.idx', 'y']]
bank_data.head(5)
bank_data['default'] = bank_data['default'].map({'no' : 0, 'yes' : 1, 'unknown' : 0})
X = bank_data.drop('y', axis = 1)
y = bank_data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
row = [[1, 57, 0, 93.994]]
y_pred = rf.predict(row)
print('Prediction', y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy : ", accuracy)
accuracy = accuracy_score(y_test, y_pred)
Input : bank.csv
age,default,cons.price.idx,cons.conf.idx,y
56,no,93.994,36.4,no
57,unknown,93.994,36.4,yes
37,no,93.994,36.4,no
40,no,93.994,36.4,no
56,no,93.994,36.4,no
Output :
Prediction ['no']
Accuracy : 1.0
19. Agglomerative Clustering
Program :
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(x, y))
hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean',
linkage='ward')
labels = hierarchical_cluster.fit_predict(data)
plt.scatter(x, y, c=labels)
plt.show()
linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.show()
print(labels)
Output : [0 0 1 0 0 1 1 0 1 1]
