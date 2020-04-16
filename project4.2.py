"""
卷烟投放二阶段
通过30挡零售户上期某品规投放数据，进行调整，并生成该品规各挡位下期投放策略¶
"""

'''
@Description: 投放模型二阶段 单品规DEMO 
@Version: V0.2
@Date: 2019-12-30 17:12:05
@LastEditors  : bruce
@LastEditTime : 2020-01-01 19:38:21
@note:
    订足面 dzm 0.5 反向
    订足率 dzl 0.5 正向
    订单满足率 ddmz 0.8 反向
    订购率 dgl 0.5 正向
    数据读取：品规_月份_三率一面表
'''

1.加载所需包
import copy
import math
import pandas as pd
import tkinter as tk  
import tkinter.messagebox
import os

2.函数封装
def get_guide(path):
    '''
    @description: 由路径(导入初始数据集)得到三率一面、本期策略等指导数据
    @param：path
    @return: {本期策略,三率一面,品规名,本期订购总量, 各组总户数}
    '''    
    dt = pd.read_csv(path, sep=',', encoding='gbk')
    strategy = list(dt['strategy']) # 本期投放策略strategy
    item = dt['item'][1] # 品规名
    dzm = list(dt['dzm'])
    dzl = list(dt['dzl'])
    ddmz = list(dt['ddmz'])
    dgl = list(dt['dgl'])
    dg = dt['dg_qty'] #各档订购量
    aimnow = sum(dg) #本期订购总量
    number = dt['group_num'] #各组总户数

    return {'strategy': strategy,
            'dzm': dzm, 'dzl': dzl,
            'ddmz': ddmz, 'dgl': dgl,
            'item': item,
            'aimnow': aimnow,
            'number': number}


def get_next_strategy(aim, guide, this_strategy):
    '''
    @description: 指导数据 in dict --> 下期策略 in list
    @param {aim:下期销量目标; guide: 三率一面; strategy: 本期策略} 
    @return: 下期策略 in list
    '''
    strategy = copy.copy(this_strategy) # 浅拷贝,否则相当于传址调用, 会修改guide_data['strategy']
    obj = aim - guide['aimnow'] # 订购量待调整量
    number_c = list(round(guide['number']*guide['dgl'])) #实际订购户数

    imf = [] #品规信息量
    for i in range(0, 30):
        t = math.log(2*(30-i))-2
        imf.append(t)
    w = [3, 3, 1, 0.1] #三率一面的权重

    #计算得分
    score = []
    for i in range(0, 30):
        temp = (w[0]*(0.5-guide['dzm'][i])+w[1]*(guide['dzl'][i]-0.5)
                +w[2]*(0.8-guide['ddmz'][i])+w[3]*(guide['dgl'][i]-0.5))-0.12*i
        score.append(temp)
    mean = 1/30*sum(score)
    for i in range(0, 30):
        score[i] = (score[i]-mean)*(1-0.02*i)

    if obj >= 0:
        delta = 1.25 * obj
        for i in range(0, 30):
            while score[i] < -0.8:
               if strategy[i] <= 0:
                   break
               strategy[i] -= 1
               delta += number_c[i]
               score[i] += 0.3
        while delta > 0:
            t = score.index(max(score))
            strategy[t] += 1
            score[t] -= 0.3
            delta -= number_c[i]
    else:
        delta = 0.8 * obj
        for i in range(0, 30):
            while score[i] > 0.8:
               strategy[i] += 1
               delta -= number_c[i]
               score[i] -= 0.3
        n = 1
        while delta < 0 and n <= 30:
            n += 1
            t = score.index(min(score))
            if strategy[t] > 0:
                strategy[t] -= 1
                score[t] += 0.3
                delta += number_c[i]
            else:
                score[t] += 10000  # 就是加完了就给他一个很大的分数, 让他接下来不会再得到品规了

    return strategy



def hit_me():
    '''
    @description: 按钮"确定"功能函数
    @return: 返回和打印下期策略
    '''
    aim_num = int(e2.get()) # 从输入框读取下期销量目标
    next_strategy = get_next_strategy(aim_num, guide_data, guide_data['strategy']) # 生成下期投放策略
    global strategy
    strategy = next_strategy # 每次点击确定，生成的下期投放策略都会赋值到全局变量strategy中
    
    text0 = '当前计划投放量：'+str(aim_num)+'  (单位：/条)'+'\n\n\n'
    for i in range(0, 30):
        if (30-i) < 10:
            text0 += '0'
        text0 += str(30-i)
        text0 += '档：'
        text0 += str(next_strategy[i])
        text0 += '      '
        if (i+1) % 5 == 0:
            text0 += '\n'
    text1 = str(text0)
    var.set(text1)


def hit_save():
    '''
    @description: 按钮"一键导出CSV"功能函数
    @return: 导出csv文件
    '''
    if strategy == []:
        ''''''
        pass
    dw = []
    for i in range(0, 30):
        dww = str(30-i)+'档'
        dw.append(dww)
    strt = pd.DataFrame(dw,columns=['档位'])
    strt['投放上限'] = pd.DataFrame(strategy)
    strt.to_csv('Result.csv', encoding = 'ANSI')
    tkinter.messagebox.showinfo(title='提示', message='导出成功！'+str(os.getcwd()))
    
    
3.实际调用
path =  'suyan.csv'  #地址可自行更改
guide_data = get_guide(path) # 读取品规数据
strategy = [] # 全局策略存储变量
# print(guide_data)
# aim_num = 5000
# res = get_next_strategy(aim_num, guide_data, guide_data['strategy'])
# print(res)
# exit()


#建立窗口windows
window = tk.Tk()
window.title('中烟自动化投放系统-2')
window.geometry('1060x720')  # 这里的乘是小x
# 前置欢迎语
l = tk.Label(window, text='欢迎来到单品规自动化投放系统！\n当前品规: '+guide_data['item'], bg='#CD7F32', font=('Arial', 20), width=50, height=3)
l.pack()    # Label内容content区域放置位置，自动调节尺寸
# c = tk.Label(window, text='您选择的数量是：', font=('Arial', 20), width=50, height=3)
# c.place(x=280,y=200)
#输入文本框
word1 = tk.Label(window, text='请输入您想要投放的数量：（单位：/条）', font=('Arial', 16), width=40, height=2)
word1.place(x=320, y=150)
e2 = tk.Entry(window, text='     ', show=None, font=('Arial', 14))  # 显示成明文形式
e2.place(x=420, y=200)
# 计算按钮
b = tk.Button(window, text='确 定', font=('Arial', 15), width=10, height=1, command=hit_me)
b.place(x=470, y=240)

# # 数量选择尺度滑条，长度500字符，从1000开始3000结束，以100为刻度，精度为1，触发调用print_selection函数
# s = tk.Scale(window, label='拖动滑块选择数量', from_=1000, to=3000, orient=tk.HORIZONTAL, length=500, showvalue=0,tickinterval=500, resolution=100, command=print_selection)
# s.place(x=280,y=200)

#输出界面
var = tk.StringVar()
output = tk.Label(window, textvariable=var, bg='#CDCDCD', fg='#00009C', font=('Arial', 18), width=68, height=12)
output.place(x=60, y=320)
b2 = tk.Button(window, text='一键导出CSV文件', font=('Arial', 15), width=15, height=1, command=hit_save)
b2.place(x=800, y=670)
window.mainloop()


