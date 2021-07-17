#加一个文件夹

import time ,os #获取日期
#coding:utf-8
import matplotlib.pyplot as plt

time1=time.strftime('%Y-%m-%d')

sv_path='pre_data/'+time1

os.makedirs(sv_path,exist_ok=True)

plt.savefig(f'{sv_path}/predict%d.pdf'%i)#保存文件在指定文件夹下很方便

plt.close()

