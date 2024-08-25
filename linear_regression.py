import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = 0
        self.c = 0

    def predict(self, X):
        return self.m * X + self.c

    def cost_function(self, X, y):
        N = len(y)
        return (1/(2*N)) * np.sum((y - self.predict(X)) ** 2)

    def gradient_descent(self, X, y):
        N = len(y)
        for i in range(self.iterations):
            predictions = self.predict(X)
            dm = (-2/N) * np.sum(X * (y - predictions))
            dc = (-2/N) * np.sum(y - predictions)
            self.m = self.m - self.learning_rate * dm
            self.c = self.c - self.learning_rate * dc
            if i % 100 == 0:  # Print cost every 100 iterations
                cost = self.cost_function(X, y)
                print(f"Iteration {i}: Cost {cost}, m {self.m}, c {self.c}")

    def fit(self, X, y):
        self.gradient_descent(X, y)

    def plot(self, X, y):
        predictions = self.predict(X)
        plt.scatter(X, y, color='blue')
        plt.plot(X, predictions, color='red')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.title('Linear Regression Fit')
        plt.show()


