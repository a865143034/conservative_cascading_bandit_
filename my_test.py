#coding:utf-8
from env import *
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import pro_movielen


u_0=0.7

def takeSecond(elem):
    return elem[1]

def oracle(U):
    A=[]
    for i in range(len(U)):
        A.append([i,U[i]])
    A.sort(reverse=True,key=takeSecond)
    ans=[]
    for i in range(K):
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

# def f_base(A,w):
#     if A==0:
#         return u_0
#     sum=0
#     for i in range(len(A)):
#         tmp=1
#         for j in range(i):
#             tmp=tmp*(1-w[A[j]])
#         tmp*=w[A[i]]
#         sum+=tmp
#     return sum


def c_3_ucb():
    reward=[]
    regret=[]
    epsilon=0.2
    f=open('file/cascading.txt', 'w')
    n=40000
    theta_hat=np.zeros(d).reshape(-1,1)
    beta=1
    delta=0.1
    lamda=1
    V=lamda*np.eye(d)
    X=[]
    Y=[]
    got_theta_star()
    start=time.time()
    sum_rew=0
    n1=0
    n2=0
    for t in range(n):
        if((t+1)%200==0):
            print('now turn:'+str(t))
            end=time.time()
            print(end-start)

        x=obtain_x()
        U=[]
        L=[]
        for i in x:
            front=np.dot(i.reshape(1,-1),theta_hat)
            V_ni=np.linalg.inv(V)
            i=i.reshape(-1,1)
            i_ni=i.reshape(1,-1)
            behind=beta*np.sqrt(i_ni.dot(V_ni).dot(i))
            tmp=min(np.asscalar(front+behind),1)
            tmp2 = max(np.asscalar(front-behind), 0)
            U.append(tmp)
            L.append(tmp2)

        A_t=oracle(U)
        idx,w=pull_(A_t)


        theta_star_1=get_theta_star()
        opt_w=[]
        for i in x:
            opt_w.append(theta_star_1.dot(i))
        f1=f_(A_t,opt_w)
        f2=f_([0,1,2,3],sorted(opt_w,reverse=True))
        f.write(str(f1)+' '+str(f2-f1)+'\n')
        sum_rew+=f1
        # print('-----------')
        # print(sum_rew)
        # print((1-epsilon)*u_0*(t+1))

        # if sum_rew>=(1-epsilon)*u_0*(t+1):
        #     n1+=1
        # else:
        #     n2+=1


        #assert 1==0
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
        #print(tm1.shape)
        theta_hat=tm1_ni.dot(np.array(X).T).dot(np.array(Y).reshape(-1,1))
        #print(theta_hat.shape)
        # o_t,w_t=pull_(A_t)

        #beta
        det=np.linalg.det(V)
        beta=np.sqrt(math.log(det/(lamda**d * delta**2)))+np.sqrt(lamda)

        if (t+1)%1000==0:
            # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            # print(n1)
            # print(n2)
            f.flush()
    f.write(str(n1)+' '+str(n2)+'\n')
    f.close()




#c_3_ucb()





def process_reward(R):
    for i in range(len(R)):
        if i!=0:
            R[i]=(R[i-1]*i+R[i])/(i+1)
    return R


def c_4_ucb(epsilon):
    reward=[]
    regret=[]
    u_0=0.7
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
    #A=[]#每个t的action
    all_x=[]
    all_L=[]
    got_theta_star()
    n1=0
    n2=0
    #f=open('conse_without.txt','w')
    start=time.time()
    for t in range(n):
        if (t+1)%200==0:
            print('now turn:'+str(t))
            end=time.time()
            print(end-start)
        x=obtain_x()
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
        #choose optimistic action
        A_t=oracle(U)
        B_t=A_t

        if u_0>f_(A_t,U):
            B_t=0

        if B_t!=0:
            tmp_x = []
            for i in A_t:
                tmp_x.append(x[i])
            tmp_L = []
            for i in A_t:
                tmp_L.append(L[i])

            all_x.append(tmp_x)
            all_L.append(tmp_L)
        else:
            all_L.append(0)
            all_x.append(0)
        ###conservative judge
        ####judge有问题，比例太有问题了
        for i in N_t:
            for k in range(4):
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
            s1+=f_([0,1,2,3],all_L[i])
        phi_t=len(D_t)*u_0+f_(B_t,L)+s1
        ###太整齐了，1：2的比例，n1,n2，3:7
        flag=False
        if phi_t >=(1-epsilon)*(t+1)*u_0 and B_t!=0:
            flag=True
            #print('A')
            n1+=1
            N_t.append(t)
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
            #print(t1)
            theta_hat = tm1_ni.dot(np.array(X).T).dot(np.array(Y).reshape(-1,1))
            #print(theta_hat)
            #print(tm1_ni)
            # beta
            det = np.linalg.det(V)
            beta = np.sqrt(math.log(det / (lamda ** d * delta ** 2))) + np.sqrt(lamda)
        else:
            n2+=1
            D_t.append(t)
        ###update reward和regret
        opt_w=[]
        theta_star_1=get_theta_star()
        for i in x:
            opt_w.append(theta_star_1.dot(i))
        f2=f_([0,1,2,3],sorted(opt_w,reverse=True))
        f2=max(f2,u_0)
        # print(f_([0,1,2,3],sorted(opt_w)))
        #
        # assert 1==0
        if flag==True:
            f1=f_(A_t,opt_w)
            reward.append(f1)
            regret.append(f2-f1)
            #f.write(str(f1)+' '+str(f2-f1)+'\n')
        else:
            #f.write(str(u_0)+' '+str(f2-u_0)+'\n')
            reward.append(u_0)
            regret.append(f2-u_0)

        if (t+1)%1000==0:
            print('&&&&&&')
            print(n1)
            print(n2)
            # f.flush()

    return regret
    # f.close()

def pro_reg(reg):
    sum=0
    for i in reg:
        sum+=i
    sum/=len(reg)
    return sum





def c_4_ucb_without(epsilon):
    reward=[]
    regret=[]

    #epsilon=0.2
    u_0=0.7
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
    #A=[]#每个t的action
    all_x=[]
    all_L=[]
    got_theta_star()
    n1=0
    n2=0
    hat_u_0=0


    #f=open('c_4_without.txt','w')
    start=time.time()
    for t in range(n):

        if (t+1)%200==0:
            print('now turn:'+str(t))
            end=time.time()
            print(end-start)

        x=obtain_x()
        tmp_u=np.random.normal(u_0,0.1)
        hat_u_0=(hat_u_0*t+tmp_u)/(t+1)
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
        radius=math.sqrt(2*math.log(40000)/(t+1))
        u_0_U=hat_u_0+radius
        #print('------------')
        #print(hat_u_0)
        #print(radius)
        #choose optimistic action
        A_t=oracle(U)
        B_t=A_t


        if u_0_U>f_(A_t,U):
            B_t=0

        if B_t!=0:
            tmp_x = []
            for i in A_t:
                tmp_x.append(x[i])
            tmp_L = []
            for i in A_t:
                tmp_L.append(L[i])

            all_x.append(tmp_x)
            all_L.append(tmp_L)
        else:
            all_L.append(0)
            all_x.append(0)

        ###conservative judge
        ####judge有问题，比例太有问题了


        for i in N_t:
            for k in range(4):
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
            s1+=f_([0,1,2,3],all_L[i])

        phi_t=len(D_t)*u_0_U+f_(B_t,L)+s1


        ###太整齐了，1：2的比例，n1,n2，3:7

        flag = False
        if phi_t >= (1 - epsilon) * (t + 1) * u_0_U and B_t != 0:
            flag = True
            # print('A')
            n1 += 1
            N_t.append(t)
            idx, w = pull_(B_t)

            # update
            # V
            for i in range(idx + 1):
                t1 = x[A_t[i]].reshape(-1, 1)
                t2 = t1.reshape(1, -1)
                V = V + t1.dot(t2)

            # X
            for i in range(idx + 1):
                X.append(x[A_t[i]].tolist())

            # print('&&&&&')
            # print(np.array(X).shape)
            # Y
            for i in range(idx + 1):
                Y.append(w[i])

            # print('^^^^^^')
            # print(np.array(Y).shape)

            # theta
            # print(np.array(X).T.shape)
            tm1 = np.array(X).T.dot(np.array(X)) + lamda * np.eye(d)
            tm1_ni = np.linalg.inv(tm1)

            # print(t1)
            theta_hat = tm1_ni.dot(np.array(X).T).dot(np.array(Y).reshape(-1, 1))
            # print(theta_hat)
            # print(tm1_ni)

            # beta
            det = np.linalg.det(V)
            beta = np.sqrt(math.log(det / (lamda ** d * delta ** 2))) + np.sqrt(lamda)

        else:
            n2 += 1
            D_t.append(t)

        ###update reward和regret
        opt_w = []
        theta_star_1 = get_theta_star()
        for i in x:
            opt_w.append(theta_star_1.dot(i))
        f2 = f_([0, 1, 2, 3], sorted(opt_w, reverse=True))
        f2 = max(f2, u_0)

        # print(f_([0,1,2,3],sorted(opt_w)))
        #
        # assert 1==0

        if flag == True:
            f1 = f_(A_t, opt_w)
            reward.append(f1)
            regret.append(f2 - f1)
            #f.write(str(f1)+' '+str(f2-f1)+'\n')
        else:
            #f.write(str(u_0)+' '+str(f2-u_0)+'\n')
            reward.append(u_0)
            regret.append(f2 - u_0)

        if (t + 1) % 1000 == 0:
            print('&&&&&&')
            print(n1)
            print(n2)
            #f.flush()
    #f.close()
    return regret


#c_4_ucb_without()


def epsilon_test():
    res=[]
    f=open('epsilon_test_without.txt','w')
    for i in range(51):
        print(i)
        a=i*0.02
        reg=c_4_ucb_without(a)
        ans=pro_reg(reg)
        f.write(str(a)+' '+str(ans)+'\n')
        res.append(ans)
        f.flush()
    #print(res)
    f.close()

#epsilon_test()

#c_4_ucb(0.5)

import random


# def get_200():
#     id=random.randint(1,199)
#     x,y=pro_movielen.cal_relation(id)
#
#     # iid=np.zeros(200).astype(int).tolist()
#     # for i in range(200):
#     #     iid[i]=i
#     #
#     # ans=random.sample(iid, 200)
#     # #print(ans)
#     # ans_x=[]
#     # ans_y=[]
#     # for i in ans:
#     #     ans_x.append(x[i])
#     #     ans_y.append(y[i])
#     # print(np.array(ans_x).shape)
#     # print(np.array(ans_y).shape)
#     # print(np.array(obtain_x()).shape)
#     return x,y


#get_200()

def movielen_c_4(epsilon):
    reward=[]
    regret=[]
    u_0=0.7
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
    #A=[]#每个t的action
    all_x=[]
    all_L=[]
    #got_theta_star()
    n1=0
    n2=0
    f=open('real_c_4.txt','w')
    start=time.time()
    a,b,c,D=pro_movielen.cal_relation()
    for t in range(n):
        if (t+1)%200==0:
            print('now turn:'+str(t))
            end=time.time()
            print(end-start)
        userid = random.randint(1, 99)
        x,feedback=pro_movielen.get_100(userid,a,b,c,D)
        #x_hat=pro_movielen.cal_relation(random.randint(0,))
        U=[]
        L=[]
        for i in np.array(x):
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
        #choose optimistic action
        A_t=oracle(U)
        B_t=A_t

        if u_0>f_(A_t,U):
            B_t=0

        if B_t!=0:
            tmp_x = []
            for i in A_t:
                tmp_x.append(x[i])
            tmp_L = []
            for i in A_t:
                tmp_L.append(L[i])
            all_x.append(tmp_x)
            all_L.append(tmp_L)
        else:
            all_L.append(0)
            all_x.append(0)
        ###conservative judge
        ####judge有问题，比例太有问题了
        for i in N_t:
            for k in range(4):
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
            s1+=f_([0,1,2,3],all_L[i])
        phi_t=len(D_t)*u_0+f_(B_t,L)+s1
        ###太整齐了，1：2的比例，n1,n2，3:7
        flag=False
        if phi_t >=(1-epsilon)*(t+1)*u_0 and B_t!=0:
            flag=True
            #print('A')
            n1+=1
            N_t.append(t)
            idx,w=pull2(B_t,feedback)
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
            #print(t1)
            theta_hat = tm1_ni.dot(np.array(X).T).dot(np.array(Y).reshape(-1,1))
            #print(theta_hat)
            #print(tm1_ni)
            # beta
            det = np.linalg.det(V)
            beta = np.sqrt(math.log(det / (lamda ** d * delta ** 2))) + np.sqrt(lamda)
        else:
            n2+=1
            D_t.append(t)
        ###update reward和regret
        # opt_w=[]
        # theta_star_1=get_theta_star()
        # for i in x:
        #     opt_w.append(theta_star_1.dot(i))
        # f2=f_([0,1,2,3],sorted(opt_w,reverse=True))
        # f2=max(f2,u_0)
        # print(f_([0,1,2,3],sorted(opt_w)))
        #
        # assert 1==0
        if flag==True:
            f1=f_(A_t,feedback)
            reward.append(f1)
            #regret.append(f2-f1)
            f.write(str(f1)+'\n')
        else:
            f.write(str(u_0)+'\n')
            reward.append(u_0)
            #regret.append(f2-u_0)

        if (t+1)%10==0:
            print('&&&&&&')
            print(n1)
            print(n2)
            f.flush()

    return regret
    # f.close()

#movielen_c_4(0.2)

def movielen_without(epsilon):
    reward=[]
    regret=[]
    #epsilon=0.2
    u_0=0.7
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
    #A=[]#每个t的action
    all_x=[]
    all_L=[]
    n1=0
    n2=0
    hat_u_0=0
    a,b,c,D=pro_movielen.cal_relation()
    f=open('movielen_without.txt','w')
    start=time.time()
    for t in range(n):
        if (t+1)%200==0:
            print('now turn:'+str(t))
            end=time.time()
            print(end-start)
        userid = random.randint(1, 99)
        x,feedback=pro_movielen.get_100(userid,a,b,c,D)
        tmp_u=np.random.normal(u_0,0.1)
        hat_u_0=(hat_u_0*t+tmp_u)/(t+1)
        U=[]
        L=[]
        behind=0
        for i in x:
            front=np.dot(i.reshape(1,-1),theta_hat)
            V_ni=np.linalg.inv(V)
            i=i.reshape(-1,1)
            i_ni=i.reshape(1,-1)
            behind=beta*np.sqrt(i_ni.dot(V_ni).dot(i))
            tmp = min(np.asscalar(front + behind), 1)
            tmp2=max(np.asscalar(front-behind),0)
            U.append(tmp)
            L.append(tmp2)
        radius=math.sqrt(2*math.log(40000)/(t+1))
        u_0_U=hat_u_0+radius
        #choose optimistic action
        A_t=oracle(U)
        B_t=A_t
        if u_0_U>f_(A_t,U):
            B_t=0
        if B_t!=0:
            tmp_x = []
            for i in A_t:
                tmp_x.append(x[i])
            tmp_L = []
            for i in A_t:
                tmp_L.append(L[i])
            all_x.append(tmp_x)
            all_L.append(tmp_L)
        else:
            all_L.append(0)
            all_x.append(0)
        ###conservative judge
        ####judge有问题，比例太有问题了
        for i in N_t:
            for k in range(4):
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
            s1+=f_([0,1,2,3],all_L[i])
        phi_t=len(D_t)*u_0_U+f_(B_t,L)+s1
        ###太整齐了，1：2的比例，n1,n2，3:7
        flag = False
        if phi_t >= (1 - epsilon) * (t + 1) * u_0_U and B_t != 0:
            flag = True
            # print('A')
            n1 += 1
            N_t.append(t)
            idx, w = pull2(B_t,feedback)
            # update
            # V
            for i in range(idx + 1):
                t1 = x[A_t[i]].reshape(-1, 1)
                t2 = t1.reshape(1, -1)
                V = V + t1.dot(t2)

            # X
            for i in range(idx + 1):
                X.append(x[A_t[i]].tolist())
            # Y
            for i in range(idx + 1):
                Y.append(w[i])
            # theta
            tm1 = np.array(X).T.dot(np.array(X)) + lamda * np.eye(d)
            tm1_ni = np.linalg.inv(tm1)
            theta_hat = tm1_ni.dot(np.array(X).T).dot(np.array(Y).reshape(-1, 1))
            # beta
            det = np.linalg.det(V)
            beta = np.sqrt(math.log(det / (lamda ** d * delta ** 2))) + np.sqrt(lamda)

        else:
            n2 += 1
            D_t.append(t)


        ###update reward和regret
        if flag == True:
            f1 = f_(A_t, feedback)
            reward.append(f1)
            #regret.append(f2 - f1)
            f.write(str(f1)+'\n')
        else:
            reward.append(u_0)
            #regret.append(f2 - u_0)
            f.write(str(u_0)+'\n')

        if (t + 1) % 100 == 0:
            print('&&&&&&')
            print(n1)
            print(n2)
            #f.flush()
    #f.close()
    return regret

#movielen_without(0.2)



def conservative_bandit(epsilon):
    reward=[]
    regret=[]
    u_0=0.7
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
    #A=[]#每个t的action
    all_x=[]
    all_L=[]
    #got_theta_star()
    n1=0
    n2=0
    f=open('baseline.txt','w')
    start=time.time()
    a,b,c,D=pro_movielen.cal_relation()
    for t in range(n):
        if (t+1)%200==0:
            print('now turn:'+str(t))
            end=time.time()
            print(end-start)
        userid = random.randint(1, 99)
        x,feedback=pro_movielen.get_100(userid,a,b,c,D)
        U=[]
        L=[]

        data=random.sample(feedback, 4)

        res=f_([0,1,2,3],feedback)
        f.write(str(res)+'\n')


        for i in np.array(x):
            front=np.dot(i.reshape(1,-1),theta_hat)
            V_ni=np.linalg.inv(V)
            i=i.reshape(-1,1)
            i_ni=i.reshape(1,-1)
            behind=beta*np.sqrt(i_ni.dot(V_ni).dot(i))
            tmp = min(np.asscalar(front + behind), 1)
            tmp2=max(np.asscalar(front-behind),0)
            U.append(tmp)
            L.append(tmp2)
        #choose optimistic action
        A_t=oracle(U)
        B_t=A_t
        if u_0>f_(A_t,U):
            B_t=0
        if B_t!=0:
            tmp_x = []
            for i in A_t:
                tmp_x.append(x[i])
            tmp_L = []
            for i in A_t:
                tmp_L.append(L[i])
            all_x.append(tmp_x)
            all_L.append(tmp_L)
        else:
            all_L.append(0)
            all_x.append(0)
        ###conservative judge
        ####judge有问题，比例太有问题了
        for i in N_t:
            for k in range(4):
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
            s1+=f_([0,1,2,3],all_L[i])
        phi_t=len(D_t)*u_0+f_(B_t,L)+s1
        ###太整齐了，1：2的比例，n1,n2，3:7
        flag=False
        if phi_t >=(1-epsilon)*(t+1)*u_0 and B_t!=0:
            flag=True
            #print('A')
            n1+=1
            N_t.append(t)
            idx,w=pull2(B_t,feedback)
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
            #print(t1)
            theta_hat = tm1_ni.dot(np.array(X).T).dot(np.array(Y).reshape(-1,1))
            #print(theta_hat)
            #print(tm1_ni)
            # beta
            det = np.linalg.det(V)
            beta = np.sqrt(math.log(det / (lamda ** d * delta ** 2))) + np.sqrt(lamda)
        else:
            n2+=1
            D_t.append(t)
        ###update reward和regret
        # opt_w=[]
        # theta_star_1=get_theta_star()
        # for i in x:
        #     opt_w.append(theta_star_1.dot(i))
        # f2=f_([0,1,2,3],sorted(opt_w,reverse=True))
        # f2=max(f2,u_0)
        # print(f_([0,1,2,3],sorted(opt_w)))
        #
        # assert 1==0
        if flag==True:
            f1=f_(A_t,feedback)
            reward.append(f1)
            #regret.append(f2-f1)
            f.write(str(f1)+'\n')
        else:
            f.write(str(u_0)+'\n')
            reward.append(u_0)
            #regret.append(f2-u_0)

        if (t+1)%1000==0:
            print('&&&&&&')
            print(n1)
            print(n2)
            f.flush()

    return regret