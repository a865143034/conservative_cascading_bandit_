#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np


f1=open('file/cascading.txt','r')
f2=open('file/c4_demo3.txt', 'r')
f3=open('file/c_4_without.txt', 'r')

thred=0.56
reg1=[]
reg2=[]
reg3=[]
for line in f1.readlines():
    try:
        line=line.strip().split(' ')
        #print(line[1])
        reg1.append(float(line[1]))
    except:
        continue
f1.close()
for line in f2.readlines():
    try:
        line=line.strip().split(' ')
        #print(line[1])
        reg2.append(float(line[1]))
    except:
        continue
f2.close()

for line in f3.readlines():
    try:
        line=line.strip().split(' ')
        #print(line[1])
        reg3.append(float(line[1]))
    except:
        continue
f3.close()


for i in range(len(reg1)):
    if i!=0:
        reg1[i]=reg1[i-1]+reg1[i]


for i in range(len(reg2)):
    if i!=0:
        reg2[i]=reg2[i-1]+reg2[i]

for i in range(len(reg3)):
    if i!=0:
        reg3[i]=reg3[i-1]+reg3[i]


figure, ax = plt.subplots()
plt.gcf().set_facecolor(np.ones(3))
plt.grid(linestyle='--')
plt.ylim(0, 7500)
plt.xlim(0, 40000)
plt.plot(reg2, '#054E9F', label='C^4-UCB without known baseline reward', linestyle='--', linewidth=2)
#plt.plot(reg2, color='coral', label=chr(949)+'=0.1', linestyle='--', linewidth=2)
plt.plot(reg3, color='g',label='C^4-UCB without known baseline reward', linestyle='-.', linewidth=2)
plt.plot(reg1, color='orange', label='C^3-UCB', linewidth=2)
plt.legend( loc=1)
plt.ylabel('Cumulative expected regret', fontsize=14)
plt.xlabel("Rounds", fontsize=14)

plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
# print labels
[label.set_fontname('Times New Roman') for label in labels]


plt.savefig('d2_.pdf')
plt.show()
