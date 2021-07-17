import matplotlib.pyplot as plt
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
        reg2.append(0.046)
    except:
        continue

plt.plot(reg0,reg1,label='C^4-UCB')
plt.plot(reg0,reg2,label='C^3-UCB')

plt.ylim(0, 0.35)
plt.xlim(0, 1)
plt.legend(loc=1)
plt.ylabel('average expected regret')
plt.savefig('epsilon_test.pdf',bbox_inches='tight')
plt.show()