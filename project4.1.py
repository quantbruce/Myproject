"""
卷烟投放一阶段
通过二次规划模型，确定各品规的总投放量
"""
1.加载所需包
import numpy as np
import pandas as pd
from scipy.optimize import minimize
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', None)

2.初始数据导入与变量提取
arg_source = pd.read_csv('step1.csv', sep=',') # 共71种品规
arg_source.head(8)

#参数和决策变量(price/sold/cap都是71*1的向量)
price = arg_source['单箱价格'].values
sold = arg_source['上期销售量'].values
cap = arg_source['产能'].values
sold.size

sold_obj = 2800 # 下期目标总销量(箱)
price_obj = 40000  # 下期目标平均单箱价格(结构)
C = 1 # 约束条件中的超参数

3.二次规划建模
minf=C∗(Sold_obj−∑Ii=1Sold_nexti)2+∑Ii=1((Sold_nexti−Soldi)/Soldi)2 
st 
0<=Sold_nexti<=Capi 
Sold_obj∗Price_obj∗0.9<=∑Ii=1Pricei∗Sold_nexti<=Sold_obj∗Price_obj∗1.1

# 目标函数 (x代表sold_next)
obj = lambda x: C*((sold_obj*250-x.sum())**2) + (((x - sold)/sold)**2).sum()

# 约束条件(x代表 sold_next)
cons = []
cons.append({'type': 'ineq', 'fun': lambda x: (price*x).sum() - sold_obj*price_obj*0.9})
cons.append({'type': 'ineq', 'fun': lambda x: (sold_obj*price_obj*1.1) - (price*x).sum()})
for i in range(sold.size):
    cons.append({'type': 'ineq', 'fun': lambda x: (x[i]-0)})
    cons.append({'type': 'ineq', 'fun': lambda x: (cap[i]-x[i])})

#查看cons前五项
cons[:5]

4.求解满足约束的解
sold_next=sold # 将上一期销量赋值给下一期
sol = minimize(obj, sold_next, method='TNC', constraints=cons)
sol

5.检验约束条件(附)
check = lambda x: x['fun'](sol.x) >= 0
np.array(list(map(check, cons)))

# 将计算的新列sold_next添加到初始数据arg_source右侧
arg_source['sold_next'] = pd.Series(np.round(sol.x), dtype='int')
arg_source




