import numpy as np

# create class
class MyLinearReg:
    def __init__(self, X, y):
        self.X = np.array(X) # independent variables / features
        self.y = np.array(y) # depentent / target variable

        self.beta_hat = None  # will store coefficients after fitting

        if self.X.ndim == 1: # do the reshaping now so we dont have to later in every method
            self.X = self.X.reshape(-1, 1)

        # ensure y is 1D. if not, then flatten
        if self.y.ndim > 1:
            self.y = self.y.flatten()

        # check if the number of observations in X and y match
        # this could be caused by flattening in cases where y is not a friendly shape
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must match.")
        
        # ones column
        self.X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))

    def fit(self):
        self.beta_hat = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y

    def get_coefficients(self):
        if self.beta_hat is None:
            raise ValueError("Model has not been fitted yet.")
        return self.beta_hat[1:]
    
    def get_intercept(self):
        if self.beta_hat is None:
            raise ValueError("Model has not been fitted yet.")
        return self.beta_hat[0]

    def predict(self, X_new):
        X_new = np.array(X_new)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        X_new = np.hstack((np.ones((X_new.shape[0], 1)), X_new))
        
        if self.beta_hat is None:
            raise ValueError("Model has not been fitted yet.")
        
        return X_new @ self.beta_hat