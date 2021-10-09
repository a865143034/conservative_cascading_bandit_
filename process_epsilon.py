import matplotlib.pyplot as plt
import numpy as np
f1=open('file/epsilon_test.txt','r')
reg0=[]
reg1=[]
num=0
reg2=[]
for line in f1.readlines():
    try:
        line=line.strip().split(' ')
        reg0.append(float(line[0]))
        reg1.append(float(line[1]))
        reg2.append(0.038)
    except:
        continue



figure, ax = plt.subplots()
# plt.tight_layout()
plt.gcf().set_facecolor(np.ones(3))
plt.grid(linestyle='--')
plt.ylim(0, 0.35)
plt.xlim(0, 1)
plt.plot(reg0,reg1, 'g', label='C^4-UCB',linestyle='--', linewidth=2)
plt.plot(reg0,reg2, color='#054E9F', label='C^3-UCB', linestyle='-.', linewidth=2)
plt.xlabel("Rounds", fontsize=14)
plt.ylabel("Average Expected Regret", fontsize=14)
plt.legend(fontsize=14, loc=1)

plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
# print labels
[label.set_fontname('Times New Roman') for label in labels]


#plt.savefig('epsilon_test2.pdf',bbox_inches='tight')
plt.savefig('epsilon_test2.pdf')
plt.show()