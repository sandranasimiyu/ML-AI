import numpy as np

class LinearRegression:
    def __init__(self, n_iters=1000, lr=0.001):
        self.n_iters=n_iters
        self.lr=lr
        self.weight=None
        self.bias=None

    def fit(self,X,Y):
        n_samples,n_features =X.shape
        self.weight= np.zeros(n_features)
        self.bias=0

        for _ in range(self.n_iters):
            y_predict= (X, self.weight) + self.bias

            dw = (1/n_samples) * np.dot(X,(Y-y_predict))
            db = (1/n_samples) * np.sum(Y-y_predict)

            self.weight= self.weight - self.lr*dw
            self.bias= self.bias - self.lr*db

    def predict(X,self):
         y_predict= (X, self.weight) + self.bias
         return y_predict