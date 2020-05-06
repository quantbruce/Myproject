## 项目名称：Kaggle Video Game Sales电子游戏销售分析

### 项目数据来源：https://www.kaggle.com/gregorut/videogamesales

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

#### 由上述统计结果可知：与之前对比发现，平台和游戏流派数目不变， 游戏发行商的数目由578降至576

# 4. 探索性数据分析
## 4.1 用户喜好
```python
Sales_by_genres = pd.pivot_table(df, index='Year', columns='Genre', values='Global_Sales', aggfunc=np.sum).sum().sort_values(ascending=False)
Sales_by_genres = pd.DataFrame(data=Sales_by_genres, columns={'Genre_Sales'})
Sales_by_genres.head(10)
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/11.png)
#### 由此可得，在整个历史期间中，最受玩家欢迎的游戏类型分别是动作、体育、射击类的游戏

```python
# 最近5年的各流派游戏全球销量情况
Sales_by_genres_near5 = pd.pivot_table(df, index='Year', columns='Genre', values='Global_Sales', aggfunc=np.sum).iloc[-5:,:].sum().sort_values(ascending=False)
Sales_by_genres_near5 = pd.DataFrame(Sales_by_genres_near5, columns=['Genre_Sales_near5'])
Sales_by_genres_near5.head(10)
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/12.png)

```python
pd.concat([Sales_by_genres[:10], Sales_by_genres_near5[:10]], axis=1).sort_values(by='Genre_Sales', ascending=False)
```
#### 而在最近的5年中, 最受玩家欢迎的前三游戏类型一样，只是射击游戏超越了运动类游戏，排名第二，通过进一步对比发现，总体来看，近些年排名趋势变化不大，Platform平台类游戏下降较快，具体原因未知

```python
###### 将对比结果可视化
fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(15, 7))
sns.barplot(x=Sales_by_genres.index, y='Genre_Sales', data=Sales_by_genres, ax=ax1)
sns.barplot(x=Sales_by_genres_near5.index, y='Genre_Sales_near5', data=Sales_by_genres_near5, ax=ax2)
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/13.png)

#### 可看出最近五年用户最喜爱的游戏类型依然还是动作类, 运动类和射击类地位互换了, 横板类（Platform）已经淡出视野，推测可能由于3D游戏技术的发展。最后悬疑类大降，具体原因还未知

```python
### 再分析下各游戏平台受欢迎程度变化
Sales_by_platform = pd.pivot_table(df, index='Year', columns='Platform', values='Global_Sales', aggfunc=np.sum).sum().sort_values(ascending=False)
Sales_by_platform  = pd.DataFrame(Sales_by_platform, columns=['Global_Sales'])
Sales_by_platform_near5 = pd.pivot_table(df, index='Year', columns='Platform', values='Global_Sales', aggfunc=np.sum).iloc[-5:,:].sum().sort_values(ascending=False)
Sales_by_platform_near5 = pd.DataFrame(Sales_by_platform_near5, columns=['Global_Sales'])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
sns.barplot(x=Sales_by_platform.index, y='Global_Sales', ax=ax1, data=Sales_by_platform)
sns.barplot(x=Sales_by_platform_near5.index, y='Global_Sales', ax=ax2, data=Sales_by_platform_near5)
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/14.png)
#### 总体可以看出，时代的发展直接体现在技术上的变革，大部分不支持最新游戏的老版平台都被慢慢淘汰了.
#### 近5年中，曾今的霸主PS2已经不见踪影，PS4独占鳌头，DS和3DS的情况如出一辙, 老牌平台之间被新版平台逐渐取代，
#### X360销量排名下降较快，但仍在苦苦坚守, 而Wli近5年的市场份额已跌落至谷底.

## 4.2.企业方向
```python
# Top5发行商的全球销量排名
Sales_by_pub = pd.pivot_table(index='Publisher', values='Global_Sales', aggfunc=np.sum,  data=df)
Sales_by_pub = Sales_by_pub.sort_values(by = 'Global_Sales', ascending=False)
Sales_by_pub_near5 = df[df['Year']>2013]
Sales_by_pub_near5 = pd.pivot_table(data=Sales_by_pub_near5, index='Publisher', values='Global_Sales', aggfunc=np.sum)
Sales_by_pub_near5 = Sales_by_pub_near5.sort_values(by='Global_Sales', ascending=False)
Sales_by_pub_near5[:5]
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/15.png)
#### Electronic Arts的市场发行量最高，Take-Two Interactive最低

```python
#1980-2020前5名发行商市场份额占比
plt.figure(figsize=(14,6))
fig1 = plt.subplot(121)
label1 = list(Sales_by_pub[:5].index)
explode1=[0.01 for i in range(5)] # 设定各项距离圆心n个半径
values1 = [round(j, 2) for i in Sales_by_pub[:5].values for j in i]
plt.pie(values1, explode=explode1, labels=label1, autopct='%1.1f%%') # 绘制饼图

#近10年内前5名发行商市场份额占比
fig2 = plt.subplot(122)
label2 = list(Sales_by_pub_near5[:5].index)
explode2=[0.01 for i in range(5)] # 设定各项距离圆心n个半径
values2 = [round(j, 2) for i in Sales_by_pub_near5[:5].values for j in i]
plt.pie(values2, explode=explode2, labels=label2, autopct='%1.1f%%') # 绘制饼图
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/16.png)
#### 可以看出，索尼(Sony)最近几年开始走下坡路，育碧(Ubisoft)市场份额稳步提升，任天堂(Nintendo)市场份额下降较多，但仍旧占据1/5市场
#### Take-Two interactive 后起之秀表现也颇为出色

```python
#### 分析各游戏平台的游戏数量情况
platGenre = pd.crosstab(index=df.Platform, columns=df.Genre)
platGenreTotal = platGenre.sum(axis=1).sort_values(ascending = False)
plt.figure(figsize=(10, 6))
sns.barplot(y=platGenreTotal.index, x=platGenreTotal.values, orient='h')
plt.xlabel = "The amount of games"
plt.ylabel = "Platform"
plt.show()
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/17.png)
#### 在整个历史时期，DS, PS2的游戏产量一骑绝尘，超出排名靠后的PS3, Will等接近1一倍。
#### 然而结合之前的分析的PS2近些年已经淡出主流市场，综合分析可以推测，PS2虽然游戏产量高，
#### 但是并没有找到市场玩家的兴趣点，真正受市场玩家的欢迎的游戏少之又少。

## 4.3 市场方向
```python
#绘制各区域市场销量走势图
Markets=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
top5_markets = pd.pivot_table(df, index='Year',values=Markets, aggfunc=np.sum)
fig=plt.figure(figsize=(10,6))
sns.lineplot(data = top5_markets)
plt.title('Top5区域市场发展趋势')
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/18.png)
#### 总体上看，可以看出各区域市场发展趋势一致，但是涨跌幅程度不同。
####  例如，从1995年各区域市场销量开始暴涨, 其中以北美市场涨幅最大，2005-2010间数据达到峰值, 但随后的2015年后又开始狂跌，具体原因未知

```python
# 分析Top5发行商历史销售情况
P=['Nintendo','Electronic Arts','Activision','Sony Computer Entertainment','Ubisoft']
df5PBL=df[df['Publisher'].isin(P)]
df5PBL_p=pd.pivot_table(data=df5PBL,index='Year',columns='Publisher',values='Global_Sales',aggfunc=np.sum)
df5PBL_p.plot(title='Top5发行商历史销售情况',figsize=(12,6))
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/19.png)
#### 任天堂(Nintendo)在2006年左右赢来了一次大爆发, 其他发行商在这个节点反而下降了, 
#### 2007年左右任天堂又被其他发行商迅速抢回市场份额, 在2009年左右任天堂最后一次反扑之后和其他发行商一起下滑

```python
##### 分析Top5发行商各自擅长的游戏领域和市场主导地位
top5_Pub_area = pd.pivot_table(data=df5PBL,index=['Genre','Publisher'],values=Markets,aggfunc=np.sum)
top5_Pub_area.sort_values(by=['Genre','Global_Sales'],ascending=False).head(15) 
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/20.png)
```python
# 将以上数值转化为百分比，方便理解
top5_Pub_area_pct=df5PBL_G_M_p.div(top5_Pub_area.groupby(level=0).sum()).round(2)
top5_Pub_area_pct=df5PBL_G_M_p_pct.sort_values(by=['Genre','Global_Sales'],ascending=False)
top5_Pub_area_pct.head(15)
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/21.png)
#### 总体上看，不难发现，任天堂依旧处于龙头地位，在大部分游戏领域都占比第一，EA则在运动和模拟领域站住了脚跟,
#### 射击领域暴雪一马当先, 动作领域育碧Ub和暴雪持平

```python
# 再分析近5年来，Top5发行商各自擅长的游戏领域和市场主导地位
top5_Pub_area_near5 = df[(df['Year']>2013)&(df['Publisher']).isin(P)]
top5_Pub_area_near5  = pd.pivot_table(data=top5_Pub_area_near5 , index=['Genre', 'Publisher'], values=Markets, aggfunc=np.sum)
top5_Pub_area_near5_pct = top5_Pub_area_near5.div(top5_Pub_area_near5.groupby(level=0).sum()).round(2)
top5_Pub_area_near5_pct = top5_Pub_area_near5_pct.sort_values(by=['Genre','Global_Sales'],ascending=False)
top5_Pub_area_near5_pct.head(15)
```
![image](https://github.com/quantbruce/Myproject/blob/master/graph/22.png)
#### 可以发现，任天堂(Nintendo)地位不倒, EA几乎垄断运动类型游戏,  暴雪(Activision	)在策略，射击，冒险类中比较突出, 
#### 育碧(Ubisoft)则占据了动作和新领域音乐类的大部分市场, 索尼(Sony)市场占比整体偏低, 且在各各市场分布较均衡








