---
title: sklearn-广义线性模型
date: 2023-04-29 09:31
tags: [OpenAI, sklearn, "深度学习", "机器学习"]
---

本文主要介绍一些常见线性回归的方法，所谓线性回归，是指模型的输出是输入的线性组合，即模型的输出是输入的加权和。线性回归是最简单的回归方法，也是最常用的回归方法之一。线性回归的优点是模型可解释性强，计算量小，容易实现。缺点是模型表达能力有限，只能用于解决线性问题。
线性回归的一般性数学表达式为:
$$\hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p$$
其中，$x = (x_1, ... , x_p)$为输入样本的特征向量，$w = (w_1, ... , w_p)$为模型的参数，$w_0$为模型的截距，$\hat{y}$为模型的输出，也就是预测值。

# 普通最小二乘法
Iris数据集通常以文本文件的形式存储，每个样本占一行，每个特征之间以逗号分隔，最后一个特征为类别标签。例如：
LinearRegression拟合一个带有系数 w = (w_1, ..., w_p) 的线性模型，使得数据集实际观测数据和预测数据（估计值）之间的残差平方和最小。其数学表达式为:
$$\underset{w}{min\,} {|| X w - y||_2}^2$$
> 这个函数可以看作是一个数学问题，其中 $X$ 是一个矩阵，$w$ 和 $y$ 是向量。在这个问题中，我们希望找到一个向量 $w$，使得矩阵 $X$ 与 $w$ 的乘积，最接近向量 $y$。具体来说，我们希望通过调整向量 $w$ 的值，使得方程 $Xw = y$ 中左边和右边的值之间的误差最小化。这个误差用欧几里得距离的平方来表示，即 $|| X w - y||_2^2$。
因此，这个函数的意义是为了解决一个最小化问题，找到一个能够最好地逼近给定数据的向量 $w$。这个问题在机器学习中很常见，因为我们希望通过拟合数据来构建预测模型。这个函数也被称为最小二乘法，是一种经典的线性回归方法。

LinearRegression会调用fit方法来拟合数据，即找到最佳的参数w。在拟合数据之后，模型的参数存储在coef_属性中。

接下来我们用diabetes数据集做一个简单的线性模型训练
diabetes是一个关于糖尿病的数据集， 该数据集包括442个病人的生理数据及一年以后的病情发展情况。 数据集中的特征值总共10项, 如下:
```text
    # 年龄
    # 性别
    #体质指数
    #血压
    #s1,s2,s3,s4,s4,s6  (六种血清的化验数据)
```

数据集的详细信息请参阅：
```python
from sklearn import datasets
diabetes = datasets.load_diabetes()
print(diabetes.DESCR)
```

```text
.. _diabetes_dataset:
    
    Diabetes dataset
    ----------------
    
    Ten baseline variables, age, sex, body mass index, average blood
    pressure, and six blood serum measurements were obtained for each of n =
    442 diabetes patients, as well as the response of interest, a
    quantitative measure of disease progression one year after baseline.
    
    **Data Set Characteristics:**
    
      :Number of Instances: 442
    
      :Number of Attributes: First 10 columns are numeric predictive values
    
      :Target: Column 11 is a quantitative measure of disease progression one year after baseline
    
      :Attribute Information:
          - age     age in years
          - sex
          - bmi     body mass index
          - bp      average blood pressure
          - s1      tc, total serum cholesterol
          - s2      ldl, low-density lipoproteins
          - s3      hdl, high-density lipoproteins
          - s4      tch, total cholesterol / HDL
          - s5      ltg, possibly log of serum triglycerides level
          - s6      glu, blood sugar level
    
    Note: Each of these 10 feature variables have been mean centered and scaled by the standard deviation times `n_samples` (i.e. the sum of squares of each column totals 1).
    
    Source URL:
    https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
    
    For more information see:
    Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004) "Least Angle Regression," Annals of Statistics (with discussion), 407-499.
    (https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf)
```

## 使用diabete数据集的线性模型实现
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据集
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# 使用数据集的其中一个特征值
diabetes_X = diabetes_X[:, np.newaxis, 2]

# 将数据集拆分为训练集和测试集
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# 将目标数据集也拆分为训练集和测试集
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# 构造线性模型对象
regr = linear_model.LinearRegression()

# 使用训练集数据(训练)拟合模型
regr.fit(diabetes_X_train, diabetes_y_train)

# 使用测试集中的数据预测分类
diabetes_y_pred = regr.predict(diabetes_X_test)

# 输出权重信息
print("Coefficients: \n", regr.coef_)

# 可视化
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
```

> **知识点1(newaxis):** <br/>
> np中的newaxis表示增加一个维度，比如原来是一维的，增加一个newaxis后就变成了二维的，原来是二维的，增加一个newaxis后就变成了三维的，以此类推。具体操作方式如下：<br/>
以上面的diabetes_X[:, np.newaxis, 2]为例，其中n_samples表示样本数量，n_features表示特征数量。<br/><br/>因此diabetes_X[:, np.newaxis, 2]表示将diabetes_X的第三个特征值(索引为2)作为新的维度，即将diabetes_X的shape从(n_samples, n_features)变成(n_samples, 1, n_features)。<br/><br/>np.nexaxis是一个常量，可以用来增加数组的维度。在这个case中，np.newaxis用于向数组添加一个新的轴(维度)。使用Numpy的切片函数获取全部的数据行，然后在第二个位置增加一个新的轴，最终得到一个形状为(n_samples, 1, n_features)的数组。最后，它选择所有样本的第三个特征，得到一个形状为(n_samples, 1)的数组。

![png](https://melon-note-1304191985.cos.ap-beijing.myqcloud.com/note/linear-regresion-01.png)

## 普通最小二乘法的复杂度
该方法使用 X 的奇异值分解来计算最小二乘解。如果 X 是一个形状为 (n_samples, n_features)的矩阵，设 
$$n_{samples}\geq n_{features}$$ 
则该方法的复杂度为 $$ O(n_{samples} n_{fearures}^2) $$

# 岭回归
岭岭回归是一种回归分析方法，可以帮助我们在处理数据时更准确地预测结果。你可以把它想象成一种"修正"的方法。<br/>
比如说，你正在使用线性回归来预测一个人的身高，但是数据中的自变量(例如体重和肩宽)之间存在强烈的相关性。这会让你的模型变得不太准确，因为数据之间的关系变得复杂了。
这时，你可以使用岭回归来“修正”这个问题。岭回归通过添加一个额外的“惩罚”项来限制自变量之间的相关性，使模型更加稳定和准确。这个“惩罚”项就像是在数据中添加一个小小的山岭，使得自变量之间不能太靠近，让模型更加平滑和稳健。<br/>
岭回归可以应用在很多领域，例如数据挖掘、机器学习、金融和生物统计等等。它是一种非常有用的统计工具，可以帮助我们更好地处理数据，做出更准确的预测。

岭回归的数学表达式：
$$\underset{w}{min\,} {|| X w - y||_2}^2 + \alpha {||w||_2}^2 $$

## 使用diabete数据集的岭回归代码
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据集
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# 使用数据集的其中一个特征值
diabetes_X = diabetes_X[:, np.newaxis, 2]

# 将数据集拆分为训练集和测试集
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# 将目标数据集也拆分为训练集和测试集
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# 构造岭回归模型对象
regr = linear_model.Ridge (alpha = .5)

# 使用训练集数据(训练)拟合模型
regr.fit(diabetes_X_train, diabetes_y_train)

# 使用测试集中的数据预测分类
diabetes_y_pred = regr.predict(diabetes_X_test)

# 输出权重信息
print("Coefficients: \n", regr.coef_)

# 可视化
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
```
![png](https://melon-note-1304191985.cos.ap-beijing.myqcloud.com/note/linear-regresion-02.png)

## 岭回归的复杂度
岭回归的复杂度与最小二乘法类似

# Lasso回归
Lasso回归是一种用于预测的统计学方法，通常用于探索一个或多个自变量与因变量之间的关系。Lasso回归的目标是通过限制回归系数的大小，从而防止过拟合的情况，这是一种常见的统计学问题，其中模型在训练数据上表现得很好，但在测试数据上表现不佳。<br/>
Lasso回归采用了一种称为"L1正则化"的技术，该技术可以使某些回归系数变为零，这就相当于对模型进行了特征选择。换句话说，Lasso回归可以找到最相关的自变量，从而使模型更加简单和可解释。<br/>举个例子来说，假设你想预测房价，你可以使用Lasso回归来找出最重要的自变量，如房屋面积、卧室数量、浴室数量等等。然后，你可以使用这些自变量来构建一个更简单、更可靠的预测模型。<br/>总之，Lasso回归是一种用于预测的统计学方法，通过限制回归系数的大小和特征选择来防止过拟合。这使得Lasso回归在处理高维数据和噪声数据时特别有用。

Lasso回归的数学表达式：
$$\underset{w}{min\,} { \frac{1}{2n_{samples}} ||X w - y||_2 ^ 2 + \alpha ||w||_1}$$
> 这个目标函数由两部分组成：第一部分是平方误差项，衡量预测值与真实值之间的差异，第二部分是L1正则化项，用于控制模型的复杂度。
> 
其中，$X$ 是自变量的矩阵，$w$ 是待求解的回归系数向量，$y$ 是因变量的向量，$n_{samples}$ 是样本的数量，$\alpha$ 是正则化强度的超参数。

## 使用diabete数据集的岭回归代码
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据集
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# 使用数据集的其中一个特征值
diabetes_X = diabetes_X[:, np.newaxis, 2]

# 将数据集拆分为训练集和测试集
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# 将目标数据集也拆分为训练集和测试集
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# 构造Lasso回归模型对象
regr = linear_model.Lasso(alpha = 0.1)

# 使用训练集数据(训练)拟合模型
regr.fit(diabetes_X_train, diabetes_y_train)

# 使用测试集中的数据预测分类
diabetes_y_pred = regr.predict(diabetes_X_test)

# 输出权重信息
print("Coefficients: \n", regr.coef_)

# 可视化
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
```
![png](https://melon-note-1304191985.cos.ap-beijing.myqcloud.com/note/linear-regresion-03.png)