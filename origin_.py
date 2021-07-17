#coding:utf-8
from env import *
import numpy as np
import math

import matplotlib.pyplot as plt

u_0=0.3

def takeSecond(elem):
    return elem[1]

def oracle(U):
    A=[]
    for i in range(len(U)):
        A.append([i,U[i]])
    A.sort(reverse=True,key=takeSecond)
    ans=[]
    for i in range(K):
        #print(A[i][0])
        ans.append(A[i][0])
    return ans


def f_(A,w):
    if A==0:
        return u_0
    sum=0
    for i in range(len(A)):
        tmp=1
        for j in range(i):
            tmp=tmp*(1-w[A[j]])
        tmp*=w[A[i]]
        sum+=tmp
    return sum

def c_3_ucb():
    reward=[]
    regret=[]

    n=2000
    theta_hat=np.zeros(d).reshape(-1,1)
    beta=1
    delta=0.1
    lamda=1
    V=lamda*np.eye(d)
    X=[]
    Y=[]
    got_theta_star()
    for t in range(n):
        if(t%2000==0):
            print('now turn:'+str(t))

        x=obtain_x()
        U=[]
        for i in x:
            front=np.dot(i.reshape(1,-1),theta_hat)
            V_ni=np.linalg.inv(V)
            i=i.reshape(-1,1)
            i_ni=i.reshape(1,-1)
            behind=beta*np.sqrt(i_ni.dot(V_ni).dot(i))
            tmp=min(np.asscalar(front+behind),1)
            U.append(tmp)
        print(U)

        assert 1==0
        A_t=oracle(U)
        idx,w=pull_(A_t)
        #print(w)
        #print(idx,w)
        #update
        #V
        for i in range(idx+1):
            t1=x[A_t[i]].reshape(-1,1)
            t2=t1.reshape(1,-1)
            V=V+t1.dot(t2)
        #print(V.shape)

        #X
        for i in range(idx+1):
            X.append(x[A_t[i]].tolist())

        #Y
        for i in range(idx+1):
            Y.append(w[i])

        #theta
        tm1=np.array(X).T.dot(np.array(X))+lamda*np.eye(d)
        tm1_ni=np.linalg.inv(tm1)
        theta_hat=tm1_ni.dot(np.array(X).T).dot(np.array(Y).reshape(-1,1))

        #beta
        det=np.linalg.det(V)
        beta=np.sqrt(math.log(det/(lamda**d * delta**2)))+np.sqrt(lamda)

    plt.plot(regret)
    plt.show()


c_3_ucb()







def process_reward(R):
    for i in range(len(R)):
        if i!=0:
            R[i]=(R[i-1]*i+R[i])/(i+1)
    return R




def c_4_ucb():
    reward=[]
    regret=[]

    epsilon=0.2
    u_0=0.3
    n=40000
    theta_hat=np.zeros(d).reshape(-1,1)
    beta=1
    delta=0.1
    lamda=1
    V=lamda*np.eye(d)
    X=[]
    Y=[]
    N_t=[]
    D_t=[]
    A=[]#每个t的action
    all_x=[]
    all_L=[]
    got_theta_star()
    n1=0
    n2=0

    f=open('conse_without.txt','w')

    for t in range(n):

        if t%200==0:
            print('now turn:'+str(t))

        x=obtain_x()
        all_x.append(x)
        #print(len(all_x[0]))
        #x的长度是200，也就是base arm的个数是200个
        U=[]
        L=[]

        behind=0
        for i in x:
            front=np.dot(i.reshape(1,-1),theta_hat)
            V_ni=np.linalg.inv(V)
            i=i.reshape(-1,1)
            i_ni=i.reshape(1,-1)
            behind=beta*np.sqrt(i_ni.dot(V_ni).dot(i))

            #tmp=np.asscalar(front+behind)
            tmp = min(np.asscalar(front + behind), 1)
            tmp2=max(np.asscalar(front-behind),0)
            U.append(tmp)
            L.append(tmp2)
        #if t%100==0:
        #   print(behind)

        all_L.append(L)#可以只存选取动作的L####待修改
        #choose optimistic action
        A_t=oracle(U)
        #print(A_t)
        B_t=A_t

        if u_0>f_(A_t,U):
            B_t=0


        ###conservative judge
        ####judge有问题，比例太有问题了


        for i in N_t:
            for k in A[i]:
                j=all_x[i][k]
                front = np.dot(j.reshape(1, -1), theta_hat)
                V_ni = np.linalg.inv(V)
                j = j.reshape(-1, 1)
                j_ni = j.reshape(1, -1)
                behind = beta * np.sqrt(j_ni.dot(V_ni).dot(j))
                tmp = max(np.asscalar(front - behind), 0)
                all_L[i][k]=tmp

        #L每轮的必须更新？还是可以存个和？

        s1=0
        for i in N_t:
            s1+=f_(A[i],all_L[i])

        phi_t=len(D_t)*u_0+f_(B_t,L)+s1

        # print('*******')
        # print(phi_t)
        # print((1-epsilon)*(t+1)*u_0)
        ###太整齐了，1：2的比例，n1,n2，3:7


        if phi_t >=(1-epsilon)*(t+1)*u_0 and B_t!=0:
            #print('A')
            n1+=1
            N_t.append(t)
            A.append(B_t)

            idx,w=pull_(B_t)

            #update
            #V
            for i in range(idx + 1):
                t1 = x[A_t[i]].reshape(-1, 1)
                t2 = t1.reshape(1, -1)
                V = V + t1.dot(t2)

            #X
            for i in range(idx + 1):
                X.append(x[A_t[i]].tolist())

            # print('&&&&&')
            # print(np.array(X).shape)
            #Y
            for i in range(idx + 1):
                Y.append(w[i])

            # print('^^^^^^')
            # print(np.array(Y).shape)

            # theta
            #print(np.array(X).T.shape)
            tm1 = np.array(X).T.dot(np.array(X)) + lamda * np.eye(d)
            tm1_ni = np.linalg.inv(tm1)

            t1=theta_hat

            #print(t1)
            theta_hat = tm1_ni.dot(np.array(X).T).dot(np.array(Y).reshape(-1,1))
            #print(theta_hat)
            #print(tm1_ni)
            flag=False
            for i in range(len(t1)):
                if t1[i]!=theta_hat[i]:
                    flag=True

            assert flag==True #保证成立

            # beta
            det = np.linalg.det(V)
            beta = np.sqrt(math.log(det / (lamda ** d * delta ** 2))) + np.sqrt(lamda)

            rew=f_(B_t,)
            reward.append(w[idx])
            f.write(str(w[idx])+'\n')

        else:
            #print('B')
            n2+=1
            A.append(0)
            D_t.append(t)
            reward.append(u_0)
            f.write(str(u_0) + '\n')

        if t%1000==0:
            print('&&&&&&')
            print(n1)
            print(n2)
            f.flush()

    reward=process_reward(reward)
    f.close()
    plt.plot(reward)
    plt.show()



#c_4_ucb()
