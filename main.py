import numpy as np
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)

    return X_scaled, y_onehot

def softmax_batch(x):
    x =x- np.max(x, axis=1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def tanchh(x):
    return 1-2/(np.exp(2*x)+1)

def forward(X_batch, W1, b1, W2, b2, W3, b3):
    net1 = np.dot(X_batch, W1) + b1
    out1 = tanchh(net1)       
    net2 = np.dot(out1,W2)+b2                         # use tanh
    out2 = tanchh(net2)       
    net3 = np.dot(out2,W3)+b3       
    out3 = softmax_batch(net3)
    return net1, out1, net2, out2, net3, out3

def dtanh(y):   # tanh derivative
    return 1 - y ** 2


def backward(X_batch, y_batch, W2, W3, out1, out2, out3):
    dE_dnet3 = out3 - y_batch   
    dE_dout2 = np.dot(dE_dnet3,W3.T)       
    dE_dnet2 = dtanh(out2)*dE_dout2      
    dE_dout1 = np.dot(dE_dnet2,W2.T)       
    dE_dnet1 = dtanh(out1)*dE_dout1       

    dW3 = np.dot(out2.T,dE_dnet3)       
    db3 = np.sum(dE_dnet3,axis=0)     
    db3=db3.reshape(1,3)
    dW2 = np.dot(out1.T,dE_dnet2)       
    db2 = np.sum(dE_dnet2,axis=0)       
    db2 = db2.reshape(1, 15)
    dW1 = np.dot(X_batch.T,dE_dnet1)       
    db1 = np.sum(dE_dnet1,axis=0)        
    db1 = db1.reshape(1, 20)
    return dW1, db1, dW2, db2, dW3, db3,dE_dnet3, dE_dout2, dE_dnet2, dE_dnet1



if __name__ == '__main__':
    
    X_scaled, y_onehot = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)
    input_size = X_train.shape[1]
    hidden_size1 = 20
    hidden_size2 = 15
    output_size = y_train.shape[1]
    W1 = np.random.randn(input_size, hidden_size1) * 0.01
    b1 = np.zeros((1, hidden_size1))
    W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
    b2 = np.zeros((1, hidden_size2))
    W3 = np.random.randn(hidden_size2, output_size) * 0.01
    b3 = np.zeros((1, output_size))

    learning_rate = 0.01
    num_epochs = 100

    for epoch in range(num_epochs):
        # Forward pass
        net1, out1, net2, out2, net3, out3 = forward(X_train, W1, b1, W2, b2, W3, b3)

        loss = -np.mean(np.sum(y_train * np.log(out3 + 1e-8), axis=1))
        #backward pass
        dW1, db1, dW2, db2, dW3, db3, dE_dnet3, dE_dout2, dE_dnet2, dE_dnet1 = backward(X_train, y_train, W2, W3, out1, out2, out3)

        # Update weights and biases
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs} completed , Loss: {loss:.4f}')

    print("Training complete. Final weights and biases updated.")

    # Evaluate on test set
    _, _, _, _, _, test_out = forward(X_test, W1, b1, W2, b2, W3, b3)
    test_predictions = np.argmax(test_out, axis=1)
    test_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(test_predictions == test_labels)
    print(f'Test accuracy: {accuracy:.4f}')

    '''
    Epoch 20/100 completed , Loss: 1.0978
    Epoch 40/100 completed , Loss: 0.3501
    Epoch 60/100 completed , Loss: 0.6966
    Epoch 80/100 completed , Loss: 0.0797
    Epoch 100/100 completed , Loss: 0.0837
    Training complete. Final weights and biases updated.
    Test accuracy: 0.9667

    '''

