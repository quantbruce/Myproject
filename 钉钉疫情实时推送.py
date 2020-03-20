#####################################################DingTalk atuo send message############################################################
import pandas as pd
import json
from datetime import datetime
import requests
import time
import hmac
import hashlib
import base64
from urllib import parse
from urllib.request import urlopen
import threading



#API接口
url = 'http://interface.sina.cn/news/wap/fymap2020_data.d.json'  

#抓取数据
content = urlopen(url, timeout=10).read().decode('gbk')
content = json.loads(content)
data = content['data']['list']


#从全国数据中分离出湖南省的疫情数据
data_hn = []
count = 0
for i in data:
    count += 1
    for key, value in i.items():
        if count == 6:  # 整个湖南省的各城市的信息
            data_hn.append(value)


#整理数据
data_hn = data_hn[6]
print(data_hn)
data_hunan_city = pd.DataFrame()

#遍历湖南省所有城市，将数据转化为DataFrame。
for city in data_hn:
    data_hunan_city = data_hunan_city.append(city, ignore_index=True)

#删除多余字段
del data_hunan_city['mapName']

#修改列名，规整数据。
data_hunan_city[['conNum', 'susNum', 'cureNum', 'deathNum']] = data_hunan_city[['conNum', 'susNum', 'cureNum', 'deathNum']].astype('int')
data_hunan_city.set_index('conNum', inplace=True)
data_hunan_city.sort_index(ascending=False, inplace=True)
data_hunan_city.reset_index(inplace=True)
data_hunan_city = data_hunan_city[['name', 'conNum', 'susNum', 'cureNum', 'deathNum']]
data_hunan_city = str(data_hunan_city)
# print(data_hunan_city)
# print(type(data_hunan_city))


#钉钉签名函数
def cal_timestamp_sign(secret):
    # 根据钉钉开发文档，修改推送消息的安全设置  https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq
    # 也就是根据这个方法，不只是要有robot_id, 还要有secret
    # 当前时间戳，单位是毫秒，与请求调用时间误差不能超过1小时
    # python3用int取整
    timestamp = int(round(time.time() * 1000))

    # 密钥，机器人安全设置界面，加签一栏下面显示的SEC开头的字符串
    secret_enc = bytes(secret.encode('utf-8'))
    string_to_sign = '{}\n{}'.format(timestamp, secret)
    string_to_sign_enc = bytes(string_to_sign.encode('utf-8'))
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    # 得到最终签名值
    sign = parse.quote_plus(base64.b64encode(hmac_code))

    return str(timestamp), str(sign)

#钉钉实时推送信息函数
def send_dingding_msg(content,
                      robot_id='-----------------------------------------------------------------',
                      secret='-------------------------------------------------------------------'):

    try:
        msg = {
            'msgtype': 'text',
            'text': {'content': content + '\n' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            }
        headers = {'Content-Type': 'application/json; charset=utf-8'}

        # https://oapi.dingtalk.com/robot/send?access_token=XXXXXX&timestamp=XXXXX&sign=XXX
        timestamp, sign_str = cal_timestamp_sign(secret)
        url = 'https://oapi.dingtalk.com/robot/send?access_token=' + robot_id + '&timestamp' + timestamp \
              + '&sign' + sign_str
        body = json.dumps(msg)
        response = requests.post(url, data=body, headers=headers, timeout=10)
        result = response.json()
        print(result)
        print('成功发送钉钉消息')
    except Exception as e:
        print('发送钉钉消息失败', e)


#任意指定时间段推送疫情数据
def cal_remind_time():
    now_time = datetime.now()
    if now_time.hour == 7 and now_time.minute == 00 and now_time.second == 00:
       send_dingding_msg('你好Bruce, 这是湖南疫情最新信息\n' + data_hunan_city)

    elif now_time.hour == 9 and now_time.minute == 30 and now_time.second == 00:
        send_dingding_msg('你好Bruce, 这是湖南疫情最新信息\n' + data_hunan_city)

    elif now_time.hour == 12 and now_time.minute == 00 and now_time.second == 00:
        send_dingding_msg('你好Bruce, 这是湖南疫情最新信息\n' + data_hunan_city)

    elif now_time.hour == 17 and now_time.minute == 30 and now_time.second == 00:
        send_dingding_msg('你好Bruce, 这是湖南疫情最新信息\n' + data_hunan_city)

    elif now_time.hour == 22 and now_time.minute == 00 and now_time.second == 00:
        send_dingding_msg('你好Bruce, 这是湖南疫情最新信息\n' + data_hunan_city)


# 主程序入口
if __name__ == '__main__':
    while True:
        cal_remind_time()
        time.sleep(10)
