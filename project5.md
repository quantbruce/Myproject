## 项目名称：Kaggle Video Game Sales电子游戏销售分析

### 项目来源：https://www.kaggle.com/gregorut/videogamesales

### 项目介绍：来源于vgchartz.com的游戏行业销售数据，通过进行数据预处理和分析、从用户、发行商、市场三个角度对电子游戏销售情况展开分析，并以图表可视化的方式展示研究成果.

###  数据介绍：包含游戏名称、类型、发行时间、发布者以及在全球各地的销售额数据。
### 各字段含义
#### RANK-总销售额的排名
#### Name-游戏的名字
#### Platform-游戏发布平台(即PC,PS4等)
#### Year-游戏发行的年份
#### Genre-游戏的类型
#### Publisher-游戏的出版者
#### NA_Sales -北美销售额(百万)
#### EU_Sales -欧洲销售额(百万)
#### JP_Sales -日本销售额(百万)
#### Other_Sales—世界其他地区销售额(百万)
#### Global_Sales—全球销售总额。


### 拟研究的问题
#### 用户：1. 用户喜爱的游戏类型、近期变化趋势？ 2.用户最喜爱的游戏平台、近期变化趋势？

#### 发行商：1. 前5名销售商的销售情况、近期变化趋势?  2. 各发行商最擅长的游戏领域？近期变化趋势？

#### 市场：1.游戏市场的总体发展趋势?  2.前5名发行商历史销售情况？ 3.五大市场的主导地位, 近期变化趋势？

# 1.包的导入
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import missingno as msno
% matplotlib inline
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# 画图风格
plt.style.use('fivethirtyeight')

# 中文标签的正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']

# 负号的正常显示
plt.rcParams['axes.unicode_minus'] = False
```
# 2. 描述性统计分析
```python
# 导入初始数据集
df = pd.read_csv('vgsales.csv')
df.head()
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/1.png)

```python
df.info() 
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/2.png)
 
 ```python
 df.describe().T
 ```
 ![image](https://github.com/quantbruce/Myproject/blob/master/graph/3.png)
 
#### 可以看到几大市场的游戏平均销量由高到低依次是北美市场、欧洲市场、日本市场，这三者占据了全球绝大部分市场
#### 销量最大的仍旧是北美市场，最小的是日本市场¶
#### 三大市场的方差波动也呈现出与游戏平均销量的同样趋势

```python
df.describe(include='object').T
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/4.png)
#### 由上述统计结果可知：该数据集种共涵盖有31个游戏平台, 12种游戏流派, 578个游戏发行商

# 3. 数据预处理
```python
df.shape
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/5.png)
```python
# 查看数据缺失值情况
df.info()
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/6.png)

```python
# 由以上结果可知, 仅在Year, Publisher字段存在缺失情况，其他字段均无缺失
df[df['Publisher'].isnull()|df['Year'].isnull()].shape 
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/7.png)

```python
# 获取缺失值占比情况
307/df.shape[0] 
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/8.png)

```python
# 由以上分析结果可知, 缺失值占比只有百分之1.8%.  故采用删除操作不会影响数据平衡
df.dropna(how='any', inplace=True)
# 再次查看数据集df有无大幅变动
df.describe().T
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/9.png)

```python
df.describe(include='object').T
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/10.png)









 
