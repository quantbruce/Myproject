###################################################每日更新股票数据######################################################
from urllib.request import urlopen
import pandas as pd
from datetime import datetime
import time
import re
import os
import json
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', None)


####函数：从具体某个网页上抓数据，不做深度处理(转化为数据框)
def get_content_from_internet(url, max_try_num=10, sleep_time=5):
    '''
    :param url:
    :param max_try_num:
    :param sleep_time:
    :return:
    '''
    get_success = False
    for i in range(max_try_num):
        try:
            content = urlopen(url, timeout=10).read().decode('gbk')
            get_success = True
            break
        except Exception as e:
            print('抓取数据失败, 第%d次尝试' % i)
            time.sleep(sleep_time)

    if get_success == True:
        return(content)

    else:
        raise ValueError('使用urlopen抓取网页数据不断报错，达到尝试上限，停止程序，请尽快检查问题所在')



####函数：从新浪财经获得指定股票数据(在从某个网站上爬取数据后，对爬取后的数据预处理为DataFrame格式, 最后返回的是DataFrame)
def get_today_data_from_sinajs(code_list):
    '''
    :param code_list:
    :return:
    '''
    url = 'http://hq.sinajs.cn/list=' + ','.join(code_list)
    content = get_content_from_internet(url)

    ##将数据转化为DataFrame
    content = content.strip()
    data_line = content.split('\n')
    data_line = [i.replace('var hq_str_', '').split(',') for i in data_line]
    df = pd.DataFrame(data_line)

    ## 对DataFrame进行调整
    df[0] = df[0].str.split('="')
    df['stock_code'] = df[0].str[0].str.strip()
    df['stock_name'] = df[0].str[1].str.strip()
    df['candle_end_time'] = df[30] + ' ' + df[31]
    df['candle_end_time'] = pd.to_datetime(df['candle_end_time'])
    rename_dict = {1: 'open', 2: 'pre_close', 3: 'close', 4: 'high', 5: 'low', 6: 'buy1', 7: 'sell1',
                   8: 'amount', 9: 'volume', 32: 'status'}
    ##amount单位是股，volume单位是元
    df.rename(columns=rename_dict, inplace=True)
    df['status'] = df['status'].str.strip('";')
    df = df[['stock_code', 'stock_name', 'candle_end_time', 'open', 'high', 'low', 'close', 'pre_close',
             'amount', 'volume', 'buy1', 'sell1', 'status']]
    return df



####函数：判断今天是否是交易日
def is_today_trading_day():
    '''
    :return:
    '''
    df = get_today_data_from_sinajs(code_list=['sh600000'])    # 'sz000002', 'sz300001'
    sh_date = df.iloc[0]['candle_end_time']
    return datetime.now().date() == sh_date




##### =====函数：从新浪获取所有股票的数据
def get_all_today_stock_data_from_sina_markcenter():
    '''
    :return:
    '''
    page_num = 1
    raw_url = 'http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page=%s&num=80&sort=symbol&asc=1&node=hs_a&symbol=&_s_r_a=sort'

    ####存储数据的DataFrame
    all_df = pd.DataFrame()

    df = get_today_data_from_sinajs(code_list=['sh600000'])  # 这里的sh600000
    sz_date = df.iloc[0]['candle_end_time'].date()

    while True:
        # 构建URL
        url = raw_url % (page_num)
        print('开始抓取页数: ', page_num)

        ## 抓取某一面网页的数据
        content = get_content_from_internet(url)
        # content = content.decode('gbk')

        ## 循环终止条件
        if 'null' in content:
            print('已爬取到网页的尽头，退出循环')
            break

        ## 通过正则表达式，给key加上引号
        content = re.sub(r'(?<={|,)([a-zA-Z][a-zA-Z0-9]*)(?=:)', r'"\1"', content)

        ## 将数据转化为dict格式, 再转化为DataFrame
        content = json.loads(content)
        df = pd.DataFrame(content, dtype='float')

        ## 对数据进行重命名
        rename_dict = {'symbol': '股票代码', 'name': '股票名称', 'open': '开盘价', 'high': '最高价', 'low': '最低价',
                       'trade': '收盘价', 'settlement': '前收盘价', 'volume': '成交量', 'amount': '成交额'}
        df.rename(columns=rename_dict, inplace=True)

        ## 将今天添加为交易日期
        df['交易日期'] = pd.to_datetime(sz_date)

        ## 对列进行重排列
        df = df[['股票代码', '股票名称', '交易日期', '开盘价', '最高价', '最低价', '收盘价', '前收盘价', '成交量', '成交额']]

        ## 合并数据
        all_df = all_df.append(df, ignore_index=True)

        ## 遍历翻页操作
        page_num += 1
        time.sleep(1)

    ## 将今天停盘的股票删除
    all_df = all_df[all_df['开盘价'] - 0.00001 > 0]
    all_df.reset_index(inplace=True, drop=True)

    ## 返回最后结果
    return all_df


df = get_all_today_stock_data_from_sina_markcenter()


# 将今日开盘交易的所有股票数据合并在一张表查看(如果有这个需求，可选以下代码)
# df.to_csv('C:/Users/47053/Desktop/today_stock_info.csv', index=False, encoding='gbk')  

#### 两个确定是否更新今日股票的条件
#### 大范围约束(第一层)
if is_today_trading_day() is False:
     print('今天不是交易日，不需要更新股票，退出程序')
     exit()  #如果今天不是交易日，则停止更新所有股票。


#### 小范围的约束(第二层)
if datetime.now().hour() < 16:
    print('今天股票尚未收盘，不需要更新股票数据，退出程序')


########将搜集数据存储到本地
for i in df.index:
    t = df.iloc[i:i+1, :]              
    stock_code = t.iloc[0]['股票代码']  

    #构建存储文件路径
    path = 'E:/stock/DataBase/test/' + stock_code + '.csv'
    # 
    if os.path.exists(path):
        t.to_csv(path, header=None, index=False, mode='a', encoding='gbk')
    else:
        pd.DataFrame(columns=['数据由XXX整理']).to_csv(path, index=False, encoding='gbk')
        t.to_csv(path, index=False, mode='a', encoding='gbk')
    print(stock_code)
