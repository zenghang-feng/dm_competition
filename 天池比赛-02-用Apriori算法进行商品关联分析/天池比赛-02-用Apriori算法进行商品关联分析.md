&emsp;&emsp;这篇文章是对天池比赛里面的商品关联分析案例的介绍，采用 Apriori 算法发现频繁项集，确定关联关系。
# 1、基本概念
## 1.1 关联分析相关概念
&emsp;&emsp;频繁项集和关联规则是关联分析中的两个基本概念：频繁项集（frequent item sets）是经常出现在一块的物品的集合，关联规则（association rules）暗示两种物品之间可能存在很强的关系。
&emsp;&emsp;关联分析中采用 支持度 去筛选出频繁项集：一个项集的支持度（support）被定义为数据集中包含该项集的记录所占的比例。从下边 图1 中可以得到，{豆奶}的支持度为4/5。而在5条交易记录中有3条包含{豆奶，尿布}，因此{豆奶，尿布}的支持度为3/5。支持度是针对项集来说的，因此可以定义一个最小支持度，而只保留满足最小支持度的项集。
&emsp;&emsp;关联分析中采用 可信度/置信度 去找出关联规则，可信度/置信度（confidence）是针对一条诸如{尿布} ➞ {葡萄酒}的关联规则来定义的。这条规则的可信度被定义为“支持度（{尿布，葡萄酒}）/支持度（{尿布}）”。从下边 图1 中可以看到，由于{尿布，葡萄酒}的支持度为3/5，尿布的支持度为4/5，所以“尿布 ➞ 葡萄酒”的可信度为3/4=0.75。这意味着对于包含“尿布”的所有记录，我们的规则对其中75%的记录都适用。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e68628daa21939183951f313b384ab64.jpeg)


## 1.2 Apriori 算法简介
&emsp;&emsp;Apriori 原理是说如果某个项集是频繁的，那么它的所有子集也是频繁的。对于下边 图2 给出的例子，这意味着如果{0,1}是频繁的，那么{0}、{1}也一定是频繁的。这个原理直观上并没有什么帮助，但是如果反过来看就有用了，也就是说 **如果一个项集是非频繁集，那么它的所有超集也是非频繁的**。如下边的 图2 ，其中 3 是非频繁的，那么 3 的超集 0 3，1 3，2 3， 0 1 3，0 2 3，1 2 3， 0 1 2 3 也是非频繁的。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8bc36faf39aedaeaf37a84ee6bc28845.jpeg)


# 2、赛题
&emsp;&emsp;这个比赛的题目来自于天池的教学赛（https://tianchi.aliyun.com/competition/entrance/531891/information），赛题以购物篮分析为背景，要求选手对品牌的历史订单数据，挖掘频繁项集与关联规则。通过这道赛题，鼓励学习者利用订单数据，为企业提供销售策略，产品关联组合，为企业提升销量的同时，也为消费者提供更适合的商品推荐。
## 2.1 数据分析
&emsp;&emsp;数据源包括4张表：order.csv，product.csv，customer.csv，date.csv ，分别为订单表，产品表，客户表，日期表。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4f72095c4d7d41e78f29643e94452e72.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/02ea01a0f59efd32485359e913bd3f35.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fcca9d6b107ec6ba805681b54bd33483.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fa4539f96e20cd27d5f2e3acecf4fca5.png)
&emsp;&emsp;本文里面只用到了订单表，具体对数据处理的思路是：首先把订单表里面全部的客户ID提取出来，然后筛选出各个客户ID对应的商品，每个客户ID对应的全部商品作为数据集中的一条记录。代码如下：

```python
import pandas as pd

#df_customer = pd.read_csv('customer.csv', encoding='gb18030')
#df_date = pd.read_csv('date.csv', encoding='gb18030')
df_order = pd.read_csv('order.csv', encoding='gb18030')	# 订单表
#df_product = pd.read_csv('product.csv', encoding='gb18030')

cus =list(df_order['客户ID'].drop_duplicates())  		# 提取出订单表里面涉及的客户ID
dit = {}												# 存放每个客户购买的商品
for i in cus:
    dit[i] = set(df_order[df_order['客户ID'] == i]['产品名称'])
```
&emsp;&emsp;处理后的数据集如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6ea03e9026c47ad26c1e9bae4312d7ef.png)


## 2.2 发现频繁项集
&emsp;&emsp;这里使用 mlxtend 模块（可以使用pip在命令行直接安装）里面的 Apriori 算法，在使用这个模块的时候，输入的初始数据需要是二维数组，所以要对上一步的数据进行一下转换，代码如下：

```python
res = []    
for k in dit:
    res.append(list(dit[k]))
```
&emsp;&emsp;然后还需要把二维数编码成 Apriori 要求的输入格式，代码如下：

```python
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(res).transform(res)
df = pd.DataFrame(te_ary, columns=te.columns_)
```

&emsp;&emsp;然后就可以寻找出输入数据里面的频繁项集，Apriori 算法里面的支持度可以看作一个超参数，这里设置的支持度是0.05。具体代码如下：

```python
from mlxtend.frequent_patterns import apriori
result = apriori(df, min_support=0.3, use_colnames=True)     # 支持度大小需要调整一下
```

&emsp;&emsp;得到的频繁项集如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/209de16eb5b6f271f13d81f3c5fab0e0.png)


## 2.3 发现关联规则
&emsp;&emsp;然后根据上一步得到的频繁项集去发现不同商品之间的关联规则，这里设置的置信度为0.5，代码如下：

```python
from mlxtend.frequent_patterns import association_rules
a_r = association_rules(result, metric='confidence', min_threshold=0.5)
```

&emsp;&emsp;得到的关联规则如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/59ddcf5766525c8a5b4762a72bcd0601.png)


参考：
https://item.jd.com/11242112.html
http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/#association_rules
