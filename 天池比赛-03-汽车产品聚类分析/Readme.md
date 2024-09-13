&emsp;&emsp;这篇文章的内容来自于天池的数据科学比赛，主要对汽车产品聚类分析。
&emsp;&emsp;赛题链接：https://tianchi.aliyun.com/competition/entrance/531892/introduction
# 1、赛题和数据
## 1.1 赛题
&emsp;&emsp;赛题以竞品分析为背景，通过数据的聚类，为汽车提供聚类分类。对于指定的车型，可以通过聚类分析找到其竞品车型。通过这道赛题，鼓励学习者利用车型数据，进行车型画像的分析，为产品的定位，竞品分析提供数据决策。
## 1.2 数据
&emsp;&emsp;数据源：car_price.csv，数据包括了205款车的26个字段
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6e51f2e8cc691ac6dd30f5f65dbb8b9c.png)
## 1.3 赛题任务
&emsp;&emsp;选手需要对该汽车数据进行聚类分析，并找到vokswagen汽车的相应竞品。（聚类分析是常用的数据分析方法之一，不仅可以帮助我们对用户进行分组，还可以帮我们对产品进行分组（比如竞品分析） 这里的聚类个数选手可以根据数据集的特点自己指定，并说明聚类的依据）

# 2、解决方案
## 2.1 思路
&emsp;&emsp;这里题目要求是找出 vokswagen 车型的竞品（这里有可能是数据中书写错误，应该是大众车型的一种型号），关于竞品的一般解释是竞争对手的产品，这里理解为面向相同用户群体、在功能价格等方面具有一定相似性的产品，例如h5的竞品，如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d3f453c62d56e801c31d720fa3e116cd.png)

&emsp;&emsp;因此，采用聚类的方法找出 vokswagen 车型的竞品，本文使用的是Kmeans聚类，是一种采用距离进行相似度度量、进而聚类的方法。
## 2.2 数据预处理
&emsp;&emsp;首先，读取数据，查看特征包含的信息，以及特征的取值类型，代码如下：

```python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:30:09 2022
@author: zenghang.feng
"""

import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

# 读取数据，查看特征
df = pd.read_csv('car_price.csv')
df.info()

"""
RangeIndex: 205 entries, 0 to 204
Data columns (total 26 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   car_ID            205 non-null    int64  
 1   symboling         205 non-null    int64  
 2   CarName           205 non-null    object 
 3   fueltype          205 non-null    object 
 4   aspiration        205 non-null    object 
 5   doornumber        205 non-null    object 
 6   carbody           205 non-null    object 
 7   drivewheel        205 non-null    object 
 8   enginelocation    205 non-null    object 
 9   wheelbase         205 non-null    float64
 10  carlength         205 non-null    float64
 11  carwidth          205 non-null    float64
 12  carheight         205 non-null    float64
 13  curbweight        205 non-null    int64  
 14  enginetype        205 non-null    object 
 15  cylindernumber    205 non-null    object 
 16  enginesize        205 non-null    int64  
 17  fuelsystem        205 non-null    object 
 18  boreratio         205 non-null    float64
 19  stroke            205 non-null    float64
 20  compressionratio  205 non-null    float64
 21  horsepower        205 non-null    int64  
 22  peakrpm           205 non-null    int64  
 23  citympg           205 non-null    int64  
 24  highwaympg        205 non-null    int64  
 25  price             205 non-null    float64
dtypes: float64(8), int64(8), object(10)
memory usage: 41.8+ KB
"""
```
&emsp;&emsp;可以筛选出数值特征绘制数据分布图，代码如下：

```python
"""筛选 carlength, horsepower, price 绘制特征分布"""
for i in ['carlength','horsepower','price']:
    fig = sns.distplot(df[i])
    fig_save = fig.get_figure()
    fig_save.savefig('{}.png'.format(i),dpi=300)
    fig_save.clear()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/336ae33d596121d18d37ac757717ace9.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/29c676c4f53b57141c311b6a01fc5d14.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/51a222eda242e4ecbaf17aa6862d3853.png)


&emsp;&emsp;然后把非数值特征转换为数值特征，对于没有顺序的特征，采用独热编码，如果特征是有序的，可以采用序号编码，以保持特征之间的相对关系，代码如下：

```python
# 筛选出离散变量的列，查看各个列的不同取值数
col_ob = list(df.select_dtypes(include=['object']).columns)
df_object = df[col_ob]
a = df_object.nunique()

# 删除 CarName 这一列，其他非数值特征进行独热编码
df_object_s = df_object.drop(columns=['CarName'])
df_object_s2 = pd.get_dummies(df_object_s)

# 找出数值类型的特征
set_col_all = set(list(df.columns))
set_col_ob = set(col_ob)
set_col_num = set_col_all ^ set_col_ob
col_num = list(set_col_num)
df_num = df[col_num]

# 把数值特征和进行独热编码之后的特征拼接在一起，后续使用时删除 car_ID 这一列
df_cat = pd.merge(df_num, df_object_s2, left_index=True, right_index=True)
```


## 2.3 进行聚类
&emsp;&emsp;处理完数据之后，就可以对数据进行聚类了。但是在进行聚类之前，我们还需要考虑数据聚合为几类是最合适的，这里采用**轮廓系数**进行判断，轮廓系数的取值范围为[-1, 1]，轮廓系数越大聚类效果越好。
&emsp;&emsp;最直接的思路是把全部特征直接输入模型，如果采用这种方式，分析聚类的结果可以发现同一个簇里面的价格会相差很多，说明实际业务逻辑和数据的映射关系出现了偏差。实际在业务场景中，我们对不同的维度关注度是不相同的，像价格这种特征就是关注度比较高的特征，要保证不同车型的价格区分度。
&emsp;&emsp;因此我们转换一下思路，对维度进行拆解，对关注程度较高的维度先单独进行聚类，这里首先对价格进行一次聚类，相当于做一次召回，代码如下：

```python
###############################################
# 只对价格进行聚类
###############################################
# 这里只对价格聚类，不用归一化
X = df_cat[['price']]

l_k = [3,4,5,6,7,8,9,10]	# 候选的聚类数量
l_s = []					# 存放每个聚类数量下的轮廓系数值

for k in l_k:
	# 初始化方式采用 k-means++，其他超参数采用默认值
    kmeans = KMeans(init="k-means++", n_clusters=k, n_init=10, random_state=0)
    
    kmeans.fit(X)
    y = kmeans.labels_
    
    score = silhouette_score(X, y)
    l_s.append(score)

# 根据l_s中轮廓系数数值的变化，在k>6时轮廓系数下降明显，这里选择最佳聚类数量为6
# 也可以尝试其他k值，得到的最终结果会有细微差异
k_num = 6
kmeans = KMeans(init="k-means++", n_clusters=k_num, n_init=10, random_state=0)
kmeans.fit(X)
y = kmeans.labels_
    
# 单纯从价格来看，可以将结果分为3类
df_lable = pd.DataFrame(y)
df_ = pd.merge(df, df_lable, left_index=True, right_index=True)

# 筛选出和 vokswagen 同一个簇的车型
df_ = df_.rename(columns={0:'lable'})
tmp = df_[df_['CarName'].str.contains('vokswagen')]
res = df_[df_['lable'] == tmp.iloc[0,26]]
```

&emsp;&emsp;然后对上一步和 vokswagen 车型处于同一个簇的数据，用除了价格之外的其他特征再次进行聚类，代码如下：

```python
###############################################
# 筛选出价格聚类后，和目标车辆一个簇的数据，用除去
# 价格之外的全部特征再次聚类
###############################################
df_cat_2 = pd.merge(res[['car_ID']], df_cat, how='left', left_on='car_ID', right_on='car_ID')

# 这里使用了 StandardScaler 归一化方法
scaler = StandardScaler()
X_2 = scaler.fit_transform(df_cat_2.drop(columns=['car_ID','price']))

l_k_2 = [3,4,5,6,7,8,9,10]
l_s_2 = []

for k in l_k_2:
    kmeans = KMeans(init="k-means++", n_clusters=k, n_init=10, random_state=0)
    
    kmeans.fit(X_2)
    y_2 = kmeans.labels_
    
    score = silhouette_score(X_2, y_2)
    l_s_2.append(score)

print(l_s_2)
# 
```
&emsp;&emsp;分析列表 l_s_2 中轮廓系数的分值，可以观察到不同k值下的轮廓系数取值都比较小，因此考虑对特征进行降维，优化聚类的效果，这里采用PCA降维，代码如下：

```python
###############################################
# 对特征降维，保留主要特征再聚类
###############################################
from sklearn.decomposition import PCA

l_p_3 = [3,4,5,6,7,8,9,10,11,12,13,14,15]
l_k_3 = [3,4,5,6,7,8,9,10]
l_s_3 = []

best_s = 0
best_p = 0
best_k = 0
for p in l_p_3:
    pca = PCA(n_components=p)
    X_3 = pca.fit_transform(X_2)
    
    for k in l_k_3:
        kmeans = KMeans(init="k-means++", n_clusters=k, n_init=10, random_state=0)
        
        kmeans.fit(X_3)
        y_3 = kmeans.labels_
        
        score = silhouette_score(X_3, y_3)
        l_s_3.append(score)
        if score > best_s:
            best_s = score
            best_p = p
            best_k = k

# 得到 
# best_p
# best_k
```
&emsp;&emsp; 上一步可以得到 best_p（最佳降维特征数量） 、best_k（最佳聚类数量），取这两个超参数得到最终的聚类结果，代码如下：

```python

# 取 best_p best_k 得到最终的聚类结果
pca = PCA(n_components=best_p)
X_3 = pca.fit_transform(X_2)

kmeans = KMeans(init="k-means++", n_clusters=best_k, n_init=10, random_state=0)
kmeans.fit(X_3)
y_3 = kmeans.labels_

df_lable_3 = pd.DataFrame(y_3)
df_3 = pd.merge(df_cat_2, df_lable_3, left_index=True, right_index=True)
df_3 = df_3.rename(columns={0:'lable'})
df_3 = pd.merge(df, df_3[['lable', 'car_ID']], how='right', left_on='car_ID', right_on='car_ID')

# data = pd.merge(pd.DataFrame(X_3), pd.DataFrame(y_3), how='left', left_index=True, right_index=True)
# data.columns = ['f1', 'f2', 'f3', 'lable']

tmp_3 = df_3[df_3['CarName'].str.contains('vokswagen')]
res_3 = df_3[df_3['lable'] == tmp_3.iloc[0,26]]

# 筛选出和 vokswagen 同标签的车型，基本是紧凑型A级轿车
res_3[['CarName', 'lable']]
"""
41              nissan gt-r      3
75            toyota corona      3
76           toyota corolla      3
87         vokswagen rabbit      3
89     volkswagen model 111      3
92  volkswagen super beetle      3
"""

# res_3[['CarName', 'carlength', 'horsepower', 'price']]
```
&emsp;&emsp; 我们可以观察这几款车型在carlength, horsepower, price 这3个特征的取值和在分布图中的分布情况，图像如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ae89730c7fcd3a407178b248bf98f6a9.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5eba9257197604a7c37d64688090edf8.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9a39eabd6533728a77d6a7912110a290.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/051fa562e2fd52c56875ca7e1fa58a89.png)
&emsp;&emsp; 可以看到几款车型的特征是比较相近的，同理也可以对比非数值特征，以上就完成了找到 vokswagen 竞品的全部过程。还可以根据业务情况对聚类过程进行微调，得到更优的聚类效果。
