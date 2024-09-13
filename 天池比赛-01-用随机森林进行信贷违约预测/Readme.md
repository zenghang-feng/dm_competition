（文章来自2022.01.08的博客）  

&emsp;&emsp;这篇文章构建了信贷违约预测数据挖掘项目的一个baseline，这个项目来源于天池数据科学大赛，是一个二分类问题。

&emsp;&emsp;赛题链接：https://tianchi.aliyun.com/competition/entrance/531830/introduction。
# 1、赛题和数据介绍
## 1.1 赛题背景
&emsp;&emsp;赛题以金融风控中的个人信贷为背景，要求选手根据贷款申请人的数据信息预测其是否有违约的可能，以此判断是否通过此项贷款，这是一个典型的分类问题。
## 1.2 赛题数据
&emsp;&emsp;数据集中的字段含义如下：
![!\[在这里插入图片描述\](https://img-blog.csdnimg.cn/1d034e8402594ae799f0dde8409f7955.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBARGF0YVNjaWVuY2Vab25l,size_20,color_FFFFFF,t_70,g_se,x_1](https://i-blog.csdnimg.cn/blog_migrate/53acfdc3099c2502cd254fc164efeedc.png)
# 2、数据探索分析和预处理
## 2.1 数据探索分析
&emsp;&emsp;首先导入需要使用的相关模块：

```python
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
```
&emsp;&emsp;然后读取训练数据集，查看数据集中特征的数据类型和缺失情况，代码如下：

```python
df = pd.read_csv('train.csv')
df.info()
```
&emsp;&emsp;执行代码，结果如下：
```python
在这里插入代码片Data columns (total 47 columns):
 #   Column              Non-Null Count   Dtype  
---  ------              --------------   -----  
 0   id                  800000 non-null  int64  
 1   loanAmnt            800000 non-null  float64
 2   term                800000 non-null  int64  
 3   interestRate        800000 non-null  float64
 4   installment         800000 non-null  float64
 5   grade               800000 non-null  object 
 6   subGrade            800000 non-null  object 
 7   employmentTitle     799999 non-null  float64
 8   employmentLength    753201 non-null  object 
 9   homeOwnership       800000 non-null  int64  
 10  annualIncome        800000 non-null  float64
 11  verificationStatus  800000 non-null  int64  
 12  issueDate           800000 non-null  object 
 13  isDefault           800000 non-null  int64  
 14  purpose             800000 non-null  int64  
 15  postCode            799999 non-null  float64
 16  regionCode          800000 non-null  int64  
 17  dti                 799761 non-null  float64
 18  delinquency_2years  800000 non-null  float64
 19  ficoRangeLow        800000 non-null  float64
 20  ficoRangeHigh       800000 non-null  float64
 21  openAcc             800000 non-null  float64
 22  pubRec              800000 non-null  float64
 23  pubRecBankruptcies  799595 non-null  float64
 24  revolBal            800000 non-null  float64
 25  revolUtil           799469 non-null  float64
 26  totalAcc            800000 non-null  float64
 27  initialListStatus   800000 non-null  int64  
 28  applicationType     800000 non-null  int64  
 29  earliesCreditLine   800000 non-null  object 
 30  title               799999 non-null  float64
 31  policyCode          800000 non-null  float64
 32  n0                  759730 non-null  float64
 33  n1                  759730 non-null  float64
 34  n2                  759730 non-null  float64
 35  n3                  759730 non-null  float64
 36  n4                  766761 non-null  float64
 37  n5                  759730 non-null  float64
 38  n6                  759730 non-null  float64
 39  n7                  759730 non-null  float64
 40  n8                  759729 non-null  float64
 41  n9                  759730 non-null  float64
 42  n10                 766761 non-null  float64
 43  n11                 730248 non-null  float64
 44  n12                 759730 non-null  float64
 45  n13                 759730 non-null  float64
 46  n14                 759730 non-null  float64
dtypes: float64(33), int64(9), object(5)
memory usage: 286.9+ MB
```
&emsp;&emsp;可以发现 employmentLength 和 n0-n14 等特征取值存在较多的缺失情况。grade、subGrade 等特征是非数值类型。后续需要对这些情况进行处理。

&emsp;&emsp;从数据集提取前5行数据，如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d4d47810d06124af047f93b1d34c8e7d.png)
## 2.2 数据预处理
&emsp;&emsp;首先把 grade 和 subgrade 两个字段转换成数值类型，由于这里是信用等级，等级之间是有次序的，因此使用序号编码的方式，将这两个字段转换成数值，代码如下：

```python
# 对 grade	subGrade 进行序号编码
oe = OrdinalEncoder()
for i in ['grade', 'subGrade']:
    tmp = oe.fit_transform(df[i].values.reshape(-1,1))
    tmp = pd.DataFrame(tmp)
    tmp.columns = [i+'new']
    df = pd.merge(df, tmp, how='left', left_index=True, right_index=True)
```
&emsp;&emsp;然后对工作年限这个字段进行处理，提取出其中的数值，代码如下：

```python
# 提取 employmentLength 中的数值
df['employmentLength'] = df['employmentLength'].fillna('0 years')
df['employmentLength_new'] = df['employmentLength'].apply(lambda x: float(re.findall(r"\d+\.?\d*", x)[0]))
```
&emsp;&emsp;然后我们筛选出正负样本的数据，分别观察两组数据中特征取值的分布情况，代码如下：

```python
# 筛选出数值特征
df_num = df.select_dtypes(include=[int, float])
col_num = df_num.columns.tolist()
col_num.remove('policyCode')                                            # 去掉 policyCode 这个特征

# 分别筛选出标签=0和标签=1的数据
df_0 = df[df['isDefault'] == 0]
df_1 = df[df['isDefault'] == 1]

df_0 = df_0.fillna(0)
df_1 = df_1.fillna(0)

# 分析各个特征在df_0和df_1上的特征分布
k = 1
for j in col_num:
    fig,axes=plt.subplots(1,2)                                     # 创建一个1行2列的图片
    sns.distplot(df_0[j],ax=axes[0])
    sns.distplot(df_1[j],ax=axes[1])
    fig.tight_layout()                                             # 调整子图间距
    fig.savefig(str(k) + '_' + str(j) + ".png", transparent=True)  # 保存图片
    k += 1
```
&emsp;&emsp;部分特征的分布图如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3d3e7fd17344ea1adf38ea3c830db07e.png)

&emsp;&emsp;可以发现正负样本的特征差异不明显，那么构建的机器学习模型精度上限可能不是太高。另外对比正负样本的数量能看出样本的数量不均衡，负样本的数量占比较小。

# 3、构建随机森林模型和调整模型参数
## 3.1 构建随机森林分类模型
&emsp;&emsp;首先对 借贷人最早报告的信用额度开启月份 这个字段进行处理，计算这个时间到当前时间的年数，生成一个新的特征。然后从 df 中筛选出我们要使用的特征，代码如下：

```python
# 对 earliesCreditLine（借贷人最早报告的信用额度开启月份） 进行处理，计算距当前时间的年数
import datetime
df['earliesCreditLine_new'] = df['earliesCreditLine'].apply(lambda x: float(re.findall(r"\d+\.?\d*", x)[0]))
df['当前时间'] = datetime.datetime.now().year
df['信用开启年数'] = df['当前时间'] - df['earliesCreditLine_new']

# 筛选出实际使用的特征
df_2 = df
df_2 = df_2.drop(columns=['grade', 'subGrade', 'employmentLength', 'issueDate', 'earliesCreditLine', '当前时间', 'earliesCreditLine_new'])
df_2 = df_2.fillna(0)
```

&emsp;&emsp;然后构建随机森林模型，代码如下：

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 提取出特征和标签列
x = df_2.drop(columns=['isDefault'])
y = df_2['isDefault']

# 将数据划分成训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# 构建随机森林模型，这里使用的模型的默认参数，对模型训练
clf = RandomForestClassifier()
clf.fit(x_train, y_train)

# 利用训练好的模型预测测试集的结果
y_pred = clf.predict(x_test)

# 查看模型的预测效果，输出 auc 值
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
auc_val = metrics.auc(fpr, tpr)
print(auc_val)
# 输出auc:0.5292283033891222

print(metrics.accuracy_score(y_test, y_pred))
# Out[5]: 0.8036166666666666
```
&emsp;&emsp;这里打印 auc 值，得到的是0.529，相对比较低，还需要进一步优化。

## 3.2 调整模型参数
&emsp;&emsp;这里采用网格搜索调整模型参数，网格搜索对模型参数组合进行枚举，从中筛选出最优的参数组合。由于网格搜索算法的效率相对较低，所以这里只对随机森林中 树的数量、生成每一棵树用的特征数量 2个参数进行调整。并且在进行网格搜索的过程中，对参数 n_jobs 进行设置，采用多进程提升执行效率，参数的设置方式见以下代码注释。
```python
#######################################################
## 调整随机森林模型的参数
#######################################################
from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier()
# 这里只对随机森林中 树的数量、生成每一棵树用的特征数量进行调整
parameters = {'n_estimators': [50, 100], 'max_features': [5, 10]}

# 查看 CPU 的核心数
from multiprocessing import cpu_count
print(cpu_count())

'''
n_jobs is an integer, specifying the maximum number of concurrently running workers. 
If 1 is given, no joblib parallelism is used at all, which is useful for debugging. 
If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. 
For example with n_jobs=-2, all CPUs but one are used.
'''
grid_search = GridSearchCV(clf, parameters, cv=5, scoring = 'roc_auc', n_jobs=-5)

print('start+++++++++++++++++++++')
grid_search.fit(x_train, y_train)
print('end  +++++++++++++++++++++')
```

&emsp;&emsp;参数调整完成之后，可以查看最佳参数的组合,代码如下：

```python
print(grid_search.best_params_)
# Out[3]: {'max_features': 5, 'n_estimators': 100}

print(grid_search.best_score_)
# Out[4]: 0.6806999331977975
```

# 4、总结
&emsp;&emsp;以上就是对信贷违约数据预处理、构建模型分类的过程，这里还有2点可以优化：
&emsp;&emsp;1、当前的特征所能达到的分类精度比较有限，还可以基于当前数据自动构建其他一些特征，或者构建信用评分卡。
&emsp;&emsp;2、网格搜索这种调参方法效率比较低，还可以使用基于贝叶斯调参的方法。

参考链接：
https://zhuanlan.zhihu.com/p/139510947
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
