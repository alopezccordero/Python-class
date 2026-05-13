# AI and Machine Learning review
## AI
AI is the simulation of human intelligence in computer science.
- AI: Broad
- Data Science - extracting patterns from data
- Data analytics - Examining datasets - business purposes.

## Machine Learning 
### basic concepts:
 - Learning: learning from past experiences ( training data)
 - exp for machines = data
 - Performance improvement: lowering errors.

### types of machine learning:
* supervised: we give labeled examples
* unsupervised: we give unlabeled examples
* reinforcement learning: we give feedback(reward) based on performance
## Key takeaways
* AI is broader than Data science and data analytics
* machine learnig = learning from data
* supervised, unsupervised and reinforcement learning
* rule based programming fails on complex perception tasks while ML succeeds because it learns patterns from data.

# Linear regression and MSE
## Regression
<Regression> is predicting a continuous numeric value
<classification> predicting a discrete class

## Data setup.
A dataset is a matrix
rows are instances
n = number of colums. x is in rage 0 to n-2 (properties)
target is n-1. our y.
properties are called features. Data is split into training set and test set.

## Linear regression.
- <concept>: a linear model expresses the relationship between an independent variable x and dependent variable y.
- <formula>: y = ax + b
    * a is the slope
    * b is the y intercept: value of y when x = 0.
- <uses>:
    * Prediction and forecasting.
    * explaining variation.
    
In linear regression we need to find the best A and B with the MSE formula

* <MSE-Formula>
    - MSE = (1/n) * sum of (H(Xi) - yi)^2
    with numpy is easier!
    - MSE = np.mmean(y_pred - y) ** 2
## Key takeaways
 * regression predicts a continuous value, classification predicts a discrete value
 * linear model is y = ax + b if single feature. or y = wx + wx2 + wxm ... + b. multiple features
 * smaller MSE = better fit
 * errors are squared so they are penalized more heavily.
 * sum of squared errors e ^2 = sum(H(xi) - yi) ^ 2; MSE = e^2 / 2
 * easier is. H(xi) is the predicted value. yi is the actual value
 * check mse_calculation.py for more info.

 # gradient descent - multiple LR and normalization.

 ## optimization
 finding the best 2 and b that minimizes the mse is an optimizatio problem. (we use w instead of a for weight notation.)

 * cost (w, b) = 1/n sum(wxi + b - y)^2
 * cost function is the MSE expressed as a function of the parameters cost(w, b)

 * in other words is mean of the summation of ax+b(prediction) - actual value. to the square.

 ### gradient descent
 gradient descent is a way to find the minimun of a cost function by stepping downhill. we use a learning rate for gradient descent

 w new = w old - (learning rate) (partial derivation of (cost function / weight))

 gradient descent for weight is w - formula explained / weight
 gradient descent for b is b - formula eplained / b

 ## multiple linear regression
 This is when there are multiple features.

 ### normalization
  this is the practice of creating a common scale for values measured in different scales

  <minMaxNormalization>: rescale to [0, 1]:
  x' = (x - xmin)/ (xmax - xmin)

  <StandardScore>: this centers to mean 0 with standard deviation 1.
  x' = (x - mean)/ standard deviation.

  ## key takeawats:
  * Training a model is solving an optimization problem.
  * gradient descent. w = w - learningrate * partial derivvation of (cost formula / current w).
  * too small learning rate = slow convergence - too large = overshoot.
  * multiple linear regression = one weight wj per feature plus a bias.

  * normalize features before training with scales differ. usually minmax an z-scores.
# The coefficient of determination
This measures how well a regression model dxplains the variability of the data. proportion of vartiance
<formula> R^2 = 1 (SSres / SStot)
where SS tot is sum if (yi - y)^2 total variability of the data around its mean
SS res residual (unexplained variability) after fitting the modell.

* the lower the residual variability is the better fit.
* if SSres = 0. R^2 = 1. but this can be overfitting.
* R^2 = 0.0 model explains no variance. just preedict mean.
* R^2 0.0 model is worse than predicting mean

R^2 measures explanatory power, not accuracy. 

# linear classifier
## binary linear classifier.
there are two classes, 0 and 1. y = +1 or -1.
the classifier finds a value of x that separates the two classes

## regression line vs classification boundary.
* linear regression produces a line. y = wx + b
* linear classification produces a decision boundary
  - with one feature is a point.
  - with two features its a line
  - three features its a plane

in a multidimensional linear classifier we use numpy
where w = n.array([w1, w2, w2])
array of weights
and the prediction is 
return 1 if np.dot(w, x) + b > 0 else -1.

## key takeaways
* a regression line and a classification boundary are different geometric objects
* in 1d the boundary is a point. 2d is a line. 3d is a plane.
* decision rule. predicted y = +1 if w x + b > 0. else is -1.
* accuracy = correct / total
* matrix/vector form is the same equation - just easier to scale to many features.

