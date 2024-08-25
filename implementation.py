import numpy as np
import pandas as pd
from sklearn import datasets
from linear_regression import LinearRegressionFromScratch


# Create the dataset
data = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_data, y_data = data
df = pd.DataFrame(X_data, columns=['Feature'])
df['Target'] = y_data

X = df.iloc[:, 0].values
y = df.iloc[:, 1].values

# Initialize and train the model
model = LinearRegressionFromScratch(learning_rate=0.01, iterations=1000)
model.fit(X, y)

# Plot the results
model.plot(X, y)
