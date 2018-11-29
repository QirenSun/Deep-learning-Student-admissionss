import numpy as np
import pandas as pd

data=pd.read_csv('student_data.csv')
data[:10]

import matplotlib.pyplot as plt

def plot_points(data):
    X=np.array(data[['gre','gpa']])
    y=np.array(data['admit'])
    admitted=X[np.argwhere(y==1)]
    rejected=X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s=25, color='red', edgecolor='k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s=25, color='cyan', edgecolor='k')
    plt.xlabel('Test(GRE)')
    plt.ylabel('Grades(GPA)')
    
plot_points(data)
plt.show()

data_rank1=data[data['rank']==1]
data_rank2 = data[data["rank"]==2]
data_rank3 = data[data["rank"]==3]
data_rank4 = data[data["rank"]==4]

plot_points(data_rank1)
plt.title('Rank 1')
plt.show()
plot_points(data_rank2)
plt.title("Rank 2")
plt.show()
plot_points(data_rank3)
plt.title("Rank 3")
plt.show()
plot_points(data_rank4)
plt.title("Rank 4")
plt.show()


one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')],axis=1)
one_hot_data = one_hot_data.drop('rank', axis=1)
one_hot_data[:10]
#scaling the data
processed_data = one_hot_data[:]
processed_data['gre']=processed_data['gre']/800
processed_data['gpa']=processed_data['gre']/4

processed_data[:10]

#Splitting the data into Training and Testing
sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9),replace=False)
train_data,test_data=processed_data.iloc[sample],processed_data.drop(sample)

"""
print("Number of training samples is", len(train_data))
print("Number of testing samples is", len(test_data))
print(train_data[:10])
print(test_data[:10]
"""

#Splitting the data into features and targets (labels)
features = train_data.drop('admit',axis=1)
targets=train_data['admit']
features_test=test_data.drop('admit',axis=1)
targets_test=test_data['admit']

print(features[:10])
print(targets[:10])

#Training the 2-layer Neural Network
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))
def error_formula(y,output):
    return -y*np.log(output)-(1-y)*np.log(1-output)

#Backpropagate the error
def error_term_formula(x,y,output):
    return (y-output)*sigmoid_prime(x)

# Neural Network hyperparameters
epochs=1000
learnrate=0.5

def train_nn(features,target,epchos,learnrate):
# Use to same seed to make debugging easier    
    np.random.seed(42)
    
    n_records,n_features=features.shape
    last_loss=None
    
# Initialize weights    
    weights=np.random.normal(scale=1/n_features**.5,size=n_features) 
    
    for e in range(epochs):
        del_w=np.zeros(weights.shape)
        for x,y in zip(features.values,targets):
            output=sigmoid(np.dot(x,weights))
            error=error_formula(y,output)
            error_term=error_term_formula(x,y,output)
            del_w+=error_term*x
            weights+=del_w*learnrate/n_records
            
            if e % (epochs / 10) == 0:
                out = sigmoid(np.dot(features, weights))
                loss = np.mean((out - targets) ** 2)
                print("Epoch:", e)
                if last_loss and last_loss < loss:
                    print("Train loss: ", loss, "  WARNING - Loss Increasing")
                else:
                    print("Train loss: ", loss)
                    last_loss = loss
                    print("=========")
    print("Finished training!")
    print(weights)
    
    
    return weights
    
weights = train_nn(features, targets, epochs, learnrate)

#Calculating the Accuracy on the Test Data
test_out=sigmoid(np.dot(features_test,weights))
predictions= test_out > 0.5
accuracy =np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))



































