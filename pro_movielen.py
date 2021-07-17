#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import random
import collections

def cal_relation():
    f1=open('ratings.csv','r')
    f2=open('ratings.csv','r')
    movie = [0]*140000
    user = [0]*140000
    movie_new=[]
    user_new=[]
    for line in f1.readlines():
        line = line.strip().split(',')
        try:
            movie[int(line[1])]+=1
            user[int(line[0])]+=1
        except:
            continue

    for i in range(len(movie)):
        movie_new.append([movie[i],i])
    for i in range(len(user)):
        user_new.append([user[i],i])
    movie_new.sort(reverse=True)
    user_new.sort(reverse=True)
    movie_new=movie_new[:100]
    user_new=user_new[:100]
    mv=set()
    us=set()
    for i in movie_new:
        mv.add(i[1])
    for i in user_new:
        us.add(i[1])
    relation=[]
    for line in f2.readlines():
        line = line.strip().split(',')
        try:
            if int(line[0]) in us and int(line[1]) in mv:
                relation.append([int(line[0]),int(line[1])])
        except:
            continue
    num1=1
    dic1=collections.defaultdict(int)
    for i in range(len(relation)):
        if dic1[relation[i][1]]==0:
            dic1[relation[i][1]]=num1
            relation[i][1]=dic1[relation[i][1]]
            num1+=1
        else:
            relation[i][1] = dic1[relation[i][1]]
    num2=1
    dic2 = collections.defaultdict(int)
    for i in range(len(relation)):
        if dic2[relation[i][0]]==0:
            dic2[relation[i][0]]=num2
            relation[i][0]=dic2[relation[i][0]]
            num2+=1
        else:
            relation[i][0] = dic2[relation[i][0]]


    B=np.zeros([100,100])
    C = np.zeros([100, 100])
    D = np.zeros([100, 100])
    for i in relation:
        B[i[0]-1][i[1]-1]=1
    for i in range(len(B)):
        for j in range(len(B[i])):
            if B[i][j]==1:
                if random.random()>0.5:
                    C[i][j]=1
                else:
                    D[i][j]=1

    a,b,c=np.linalg.svd(C)
    return a,b,c,D



def get_100(userid,a,b,c,D):
    A1=a[userid-1].tolist()
    #print(A1)
    ans_x=[]
    for i in range(100):
        A2=c[:,i].tolist()
        ans_x.append(A1+A2)
    ans=np.array(ans_x)
    #print(ans)
    return ans,D[userid-1]

#c=cal_relation(1)
#print(c)