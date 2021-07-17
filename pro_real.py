#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np


f1=open('file/real2.txt', 'r')
f2=open('file/movielen2.txt', 'r')
# f3=open('file/u_3_test.txt', 'r')
# f4=open('file/u_4_test.txt', 'r')
# f5=open('file/u_5_test.txt', 'r')
reg1=[]
reg2=[]
reg3=[]
reg4=[]
reg5=[]
for line in f1.readlines()[:10000]:
    line=line.strip().split(' ')
    reg1.append(float(line[0]))

f1.close()
for line in f2.readlines()[:10000]:
    line=line.strip().split(' ')
    #print(line[1])
    reg2.append(float(line[0]))
f2.close()

for i in range(len(reg1)):
    if i!=0:
        reg1[i]=(reg1[i-1]*i+reg1[i])/(i+1)
for i in range(len(reg2)):
    if i!=0:
        reg2[i]=(reg2[i-1]*i+reg2[i])/(i+1)


# for i in range(len(reg1)):
#     if i!=0:
#         reg1[i]=(reg1[i-1]+reg1[i])
# for i in range(len(reg2)):
#     if i!=0:
#         reg2[i]=(reg2[i-1]+reg2[i])

# figure, ax = plt.subplots()
# # plt.tight_layout()
# plt.gcf().set_facecolor(np.ones(3))
# plt.grid(linestyle='--')
# plt.ylim(0, 0.85)
# plt.xlim(0, 40000)
#
#
# plt.plot(D,A,'#054E9F',linestyle='-.',label='Non-private',linewidth=2)
# plt.plot(D,B,color='coral',label='Laplace-LDP',linestyle='--',linewidth=2)
# plt.plot(D,C,color='m',label='Gaussian-LDP',linestyle=':',linewidth=2)
#
#
#
# plt.plot(reg1,'b',label='u_0'+'=0.2')
# plt.plot(reg2,'g',label='u_0'+'=0.5')
# plt.plot(reg3,'r',label='u_0'+'=0.7')
# plt.plot(reg4,'c',label='u_0'+'=0.9')
# plt.plot(reg5,'m',label='u_0'+'=0.95')
# plt.legend(loc=1)
# plt.ylabel('average expected regret')
# plt.savefig('u_0.pdf',bbox_inches='tight')
# plt.show()



def plot_4(reg1,reg2,reg3,reg4,reg5):

    figure, ax = plt.subplots()
    # plt.tight_layout()
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(linestyle='--')
    #plt.ylim(0.7, 0.75)
    plt.xlim(0, 10000)
    plt.plot(reg1, 'g', label='known baseline reward', linewidth=2)
    plt.plot(reg2, color='coral', label='unknown baseline reward', linestyle='--', linewidth=2)
    # plt.plot(reg3, color='#054E9F', label='u_0'+'=0.7', linestyle='-.', linewidth=2)
    # plt.plot(reg4, color='#054E9F', label='u_0'+'=0.9', linestyle=':', linewidth=2)
    # plt.plot(reg5, color='r', label='u_0'+'=0.95', linestyle='-.', linewidth=2)
    plt.ylabel("Average Reward", fontsize=14)
    plt.xlabel("Rounds", fontsize=14)
    plt.legend(fontsize=14, loc=4)

    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    # print labels
    [label.set_fontname('Times New Roman') for label in labels]

    plt.savefig('real2.pdf')
    plt.show()



plot_4(reg1,reg2,reg3,reg4,reg5)

