# PROJECT NAME: Logistic Regression with DMV Test Scores

## OVERVIEW
This project implements logistic regression to classify the results of DMV written tests (pass or fail) based on two test scores. It involves using Python libraries like NumPy, Pandas, Matplotlib, and Seaborn to visualize the data, calculate the cost function, and optimize the logistic regression parameters using gradient descent.

## TABLE OF CONTENTS
1. Installation
2. Usage
3. Features
4. Documentation
5. Credits

## INSTALLATION 

### Prerequisites
- Python 3.10.9 (this is the version used for development and testing)
- Third-party libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`

### Steps
1. Clone the repository:
    git clone https://github.com/ocampbell378/LinearRegressionWithNumPy.git

2. Install the required libraries:
    pip install -r requirements.txt

## USAGE
To run the project, use the following command:
    python main.py

## FEATURES
**Feature 1**: Data visualization using scatter plots to display the relationship between two DMV test scores and their results (pass/fail).
**Feature 2**: Standardize the scores to have zero mean and unit variance for better numerical stability during training.
**Feature 3**: Implements the logistic function for binary classification.
**Feature 4**: Defines the cost function for logistic regression and computes gradients needed for optimization.
**Feature 5**: Apply gradient descent to minimize the cost function and optimize the parameters of the logistic regression model.
**Feature 6**: Plot the decision boundary on top of the scatter plot to separate the passing and failing results.
**Feature 7**: Use the trained logistic regression model to predict the probability of passing the DMV test for new scores.

## DOCUMENTATION
### Modules and Functions
- **main.py**: Contains the primary logic for processing, visualizing data, and implementing logistic regression.
- `logistic_function(x)`: Implements the logistic (sigmoid) function to convert input values into a probability range between 0 and 1.
- `cost_function(theta, x, y)`: Computes the cost function for logistic regression and returns both the cost and the gradients for each parameter.
- `gradient_descent(x, y, theta, alpha, iterations)`: Runs gradient descent to iteratively update the logistic regression parameters `theta` by minimizing the cost function.
- `predict(theta, x)`: Uses the optimized parameters `theta` to predict if the input data (test scores) result in a pass or fail.
- Additional visualization functions are used to plot the data, decision boundary, and convergence of the cost function.

## CREDITS
- Developed by Owen Campbell
- This project was guided by the "Logistic Regression with NumPy and Python" course by Snehan Kekre on Coursera.