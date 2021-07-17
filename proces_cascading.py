import matplotlib.pyplot as plt
f1=open('file/cascading.txt','r')


reg1=[]
num=0
for line in f1.readlines():
    if num==10000:
        break
    try:
        line=line.strip().split(' ')
        #print(line[1])
        reg1.append(float(line[1]))
    except:
        continue

sum=0
for i in reg1:
    sum+=i
print(sum/10000)