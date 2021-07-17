#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

f1=open('file/c4_demo1.txt', 'r')
f2=open('file/c4_demo2.txt', 'r')
f3=open('file/c4_demo3.txt', 'r')
f4=open('file/c4_demo4.txt', 'r')
f5=open('file/c4_demo5.txt', 'r')
reg1=[]
reg2=[]
reg3=[]
reg4=[]
reg5=[]
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
for line in f4.readlines():
    try:
        line=line.strip().split(' ')
        #print(line[1])
        reg4.append(float(line[1]))
    except:
        continue
f4.close()
for line in f5.readlines():
    try:
        line=line.strip().split(' ')
        #print(line[1])
        reg5.append(float(line[1]))
    except:
        continue
f5.close()

for i in range(len(reg1)):
    if i!=0:
        reg1[i]=(reg1[i-1]*i+reg1[i])/(i+1)
for i in range(len(reg2)):
    if i!=0:
        reg2[i]=(reg2[i-1]*i+reg2[i])/(i+1)
for i in range(len(reg3)):
    if i!=0:
        reg3[i]=(reg3[i-1]*i+reg3[i])/(i+1)
for i in range(len(reg4)):
    if i!=0:
        reg4[i]=(reg4[i-1]*i+reg4[i])/(i+1)
for i in range(len(reg5)):
    if i!=0:
        reg5[i]=(reg5[i-1]*i+reg5[i])/(i+1)

# for i in range(len(reg)):
#     if i!=0:
#         reg[i]=reg[i-1]+reg[i]
# plt.ylim(0, 0.4)
# plt.xlim(0, 40000)
# plt.plot(reg1,'b',label=chr(949)+'=0.01')
# plt.plot(reg2,'g',label=chr(949)+'=0.1')
# plt.plot(reg3,'r',label=chr(949)+'=0.2')
# plt.plot(reg4,'c',label=chr(949)+'=0.5')
# plt.plot(reg5,'m',label=chr(949)+'=0.8')
# plt.legend(loc=1)
# plt.ylabel('average expected regret')
# plt.savefig('many_.pdf',bbox_inches='tight')
# plt.show()




def plot_4(reg1,reg2,reg3,reg4,reg5):

    figure, ax = plt.subplots()
    # plt.tight_layout()
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(linestyle='--')
    plt.ylim(0, 0.4)
    plt.xlim(0, 40000)
    #
    # plt.fill_between(x=np.arange(len(avg_accuracies)),
    #                  y1=avg_accuracies - sd_accuracies,
    #                  y2=avg_accuracies + sd_accuracies,
    #                  alpha=0.25
    #                  )


    plt.plot(reg2, 'g', label=chr(949)+'=0.1', linewidth=2)
    #plt.plot(reg2, color='coral', label=chr(949)+'=0.1', linestyle='--', linewidth=2)
    plt.plot(reg3, color='#054E9F', label=chr(949)+'=0.2', linestyle='-.', linewidth=2)
    plt.plot(reg4, color='#054E9F', label=chr(949)+'=0.5', linestyle=':', linewidth=2)
    plt.plot(reg5, color='r', label=chr(949)+'=0.8', linestyle='--', linewidth=2)
    plt.ylabel("Average Expected Regret", fontsize=14)
    plt.xlabel("Rounds", fontsize=14)
    plt.legend(fontsize=14, loc=1)

    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    # print labels
    [label.set_fontname('Times New Roman') for label in labels]

    plt.savefig('combi_2.pdf')
    plt.show()

plot_4(reg1,reg2,reg3,reg4,reg5)
