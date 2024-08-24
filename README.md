# Linear Regression from Scratch

This project demonstrates a simple implementation of Linear Regression from scratch in Python. The goal is to understand the fundamentals of Linear Regression by coding the algorithm without relying on high-level libraries like scikit-learn.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Linear Regression is one of the most basic algorithms in machine learning. This project implements Linear Regression using only basic Python libraries, focusing on the following key concepts:

- **Hypothesis Function**: `y = mx + c`
- **Cost Function**: Mean Squared Error (MSE)
- **Optimization Technique**: Gradient Descent

By understanding these concepts, you can gain a deeper understanding of how Linear Regression works under the hood.

## Features

- **Gradient Descent**: Implemented from scratch to optimize the model parameters.
- **Cost Function Calculation**: Tracks the cost to understand model performance over iterations.
- **Modular Design**: The code is structured using a class, making it easy to extend and reuse.
- **Data Visualization**: Includes plotting functionality to visualize the regression line and data points.

## Installation

To run this project, you'll need to have Python installed. Follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Elotmanix/linear-regression-from-scratch.git
    cd linear-regression-from-scratch
    ```

2. **Install dependencies**:

    The only dependencies are `numpy`, `pandas`, `matplotlib`, and `scikit-learn`. You can install them using pip:

    ```bash
    pip install numpy pandas matplotlib scikit-learn
    ```

## Usage

Once you've installed the dependencies, you can run the script to see the Linear Regression model in action.

1. **Run the Python script**:

    ```bash
    python linear_regression.py
    ```

   The script will train the Linear Regression model on synthetic data, print the cost at intervals, and plot the regression line.

## Examples

Here’s an example of how to use the Linear Regression class in your own projects:

```python
from linear_regression import LinearRegressionFromScratch
import numpy as np

# Generate synthetic data
X = np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

# Initialize the model
model = LinearRegressionFromScratch(learning_rate=0.01, iterations=1000)

# Train the model
model.fit(X, y)

# Plot the results
model.plot(X, y)
```
## Contributing
Contributions are welcome! If you have suggestions or improvements, please open an issue or create a pull request.
