import numpy as np

class MyLinearReg:
    def __init__(self, X, Y):
        self.X = np.array(X) # the independent variables
        self.Y = np.array(Y) # target variable

        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)

    @staticmethod
    def z(X_i):  # Z-scores for a variable set
        X_i = np.array(X_i)
        return (X_i - np.mean(X_i)) / np.std(X_i)

    @staticmethod
    def r(X_i, Y):  # correlation score for a dependent variable X_i against Y
        X_i, Y = np.array(X_i), np.array(Y)
        if len(X_i) != len(Y):
            raise ValueError("X_i and Y are arrays of different lengths")
        
        return np.sum(MyLinearReg.z(X_i) * MyLinearReg.z(Y)) / len(X_i)

    def slope(self):  # the slope vector for each orthogonal independent variable
        return np.array([MyLinearReg.r(self.X[:, i], self.Y) * np.std(self.Y) / np.std(self.X[:, i]) for i in range(self.X.shape[1])])

    def intercept(self):  # the y intercept of the linear regression model
        return np.mean(self.Y) - np.dot(self.slope(), np.mean(self.X, axis=0))
    
    def predict(self, X_new):  # make a prediction based on training data
        X_new = np.array(X_new)
        if X_new.shape[1] != self.X.shape[1]:
            raise ValueError("X_new has a different number of features than X")
        return np.dot(X_new, self.slope()) + self.intercept()
    
    def r_squared(self): # using residuals, get an R^2 value
        y_pred = self.predict(self.X)
        ss_res = np.sum((self.Y - y_pred) ** 2)
        ss_tot = np.sum((self.Y - np.mean(self.Y)) ** 2)
        return 1 - ss_res / ss_tot