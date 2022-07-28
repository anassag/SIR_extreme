# Tirex

The TIREX (Tail Inverse Regression of EXtreme response) package is a tool for dimensionality reduction in a regression context where the goal is to predict extreme values of the target. More precisely, TIREX extracts  features that are particularly relevant to predict an exceedance of the target variable  Y above a high threshold. A typical field of application is  risk management, where an important task is to identify risk factors that are linear combinations of the input and explain extreme values of the target. TIREX may be seen as an adaptation of the Sliced Inverse Regression (SIR) methods to handle the specific task  of predicting tail events. The package implements the methods proposed in [1]. 

[1] Aghbalou, A., Portier, F., Sabourin, A., & Zhou, C. (2021). Tail inverse regression for dimension reduction with extreme response. arXiv preprint arXiv:2108.01432.


The provided package is also compatible with sickit-learn `Pipeline`. 

If the number of components `n_components` set by the user is small (between 1 and 3), then we recommand to use the first order method by setting method="FO" ( default value) . If one wants to extract more than 4 features, then he can set the method to "SO" (Second order method). For more details about each method, please check our paper.

To perform Tirex, we need to set a value for the parameter `k`. If one is interested in the large values, then `k` must be small (relatively to  `n` the number of  observations). In general taking `k=sqrt(n)` will do the trick.

Calling `.fit`  will fit the model on the training data X.
.

```ruby
Tirex=TIREX(n_components=2,k=1000)
Tirex.fit(X_train)
```
Calling `.fit_transform` will fit the model and perform the dimensionality reduction task.

```ruby
Tirex=TIREX(n_components=2,k=1000)
X_train_reduce=Tirex.fit_transform(X_train)
```
Calling `.transform` will perform the dimensionality reduction task using a pre-trained model. 
```ruby
X_test_reduce=Tirex.transform(X_test)
```
