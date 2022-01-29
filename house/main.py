import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# load data
df_train = pd.read_csv('./train.csv')
df_train.head()
columns = df_train.columns

print(columns)
print(pd.value_counts(columns))
print(df_train['SalePrice'].describe()) #研究房价（target），即结果一列，会显示该列的均值 方差 标准差 四分位数 最值等信息
# pandas 是基于numpy构建的含有更高级数据结构和工具的数据分析包，提供了高效地操作大型数据集所需的工具。
# pandas有两个核心数据结构 Series和DataFrame，分别对应了一维的序列和二维的表结构。
# 而describe()函数就是返回这两个核心数据结构的统计变量。其目的在于观察这一系列数据的范围、大小、波动趋势等等，为后面的模型选择打下基础。

sns.distplot(df_train['SalePrice'])
plt.show()

# skewness and kurtosis
print("skewness: %f" % df_train['SalePrice'].skew())
print("kurtosis: %f" % df_train['SalePrice'].kurt())
# 偏度（skewness）也称为偏态、偏态系数，是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征。
# 峰度（peakedness;kurtosis）又称峰态系数。表征概率密度分布曲线在平均值处峰值高低的特征数。直观看来，峰度反映了峰部的尖度。

# scatter plot grlivarea/saleprice 研究某个特征与房价的关系
grLivArea = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[grLivArea]], axis=1)
data.plot.scatter(x=grLivArea, y='SalePrice', ylim=(0, 800000))
plt.show()

# scatter plot totalBsmtSF/saleprice 研究某个特征与房价的关系
totalBsmtSF = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[totalBsmtSF]], axis=1)
data.plot.scatter(x=totalBsmtSF, y='SalePrice', ylim=(0, 800000))
plt.show()

#box plot overallqual/saleprice #研究类别型数据与房价的分布描述
# 在这里使用的是箱型图（盒图）.
# 主要包含六个数据节点，将一组数据从大到小排列，分别计算出上边缘，上四分位数Q3，中位数，下四分位数Q1，下边缘，分散的是异常值。
# 上下边缘之间是正常数据的分布区间
overallqual = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[overallqual]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=overallqual, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.show()

#box plot yearbuilt/saleprice #研究类别型数据与房价的分布描述
yearbuilt = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[yearbuilt]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=yearbuilt, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
plt.show()

# 相关性分析
#correlation matrix相关矩阵
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()
# 相关矩阵可以用颜色和数值表示任意两个元素之间的相关性，颜色越淡，表示相关性越强，反之相关性越弱，可参考colorbar。

#saleprice correlation matrix 相关矩阵
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# 分析：两个相同元素之间的相关性是1，
# 如saleprice与saleprice，另外从图中我们看到garagecars与garagearea的相关性在0.88，
# 表明这两个元素相关性很强，因此在后面做数据分析（机器学习）时，可以只取其中一个元素特征即可。
# 相似的还有totalbsmtSF与1stFlrSF。同时，如果一个特征与saleprice之间的相关性很弱，那么这个特征可以舍去。

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show()
