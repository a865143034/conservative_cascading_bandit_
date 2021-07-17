#coding:utf-8
import numpy as np

d=200
L=100
K=4
theta_star=0
global_x=0
def pendulum():
    return np.random.uniform(-1, 1)

def uni(x):#归一化
    return x/np.sqrt(x.dot(x))

def suni(d):#产生归一化的d维向量
    x = np.random.normal(0, 1, d)
    return uni(x)


def got_theta_star():
    tmp=suni(d-1)
    theta=np.append(tmp/2,[1/2])
    global theta_star
    theta_star=theta

def get_theta_star():
    global theta_star
    #print(theta_star)
    return theta_star

def obtain_x():
    global global_x
    a=[]
    for i in range(L):
        tmp=suni(d-1)
        theta=np.append(tmp,[1])
        a.append(theta)
    a=np.array(a)
    global_x=a
    return a

def obtain_x_1():
    global global_x
    a=[]
    for i in range(L+4):
        tmp=suni(d-1)
        theta=np.append(tmp,[1])
        a.append(theta)
    a=np.array(a)
    global_x=a
    return a


def pull_(a_t):
    exp_w=[]
    w=[]
    for i in range(L):
        tmp=theta_star.dot(global_x[i])
        exp_w.append(tmp)
    for i in range(L):
        obs = np.random.binomial(1,exp_w[i])
        w.append(obs)
    exam=[]
    for i in a_t:
        exam.append(w[i])
    idx=K-1
    ans_w=[]
    for i in range(K):
        ans_w.append(exam[i])
        if exam[i]==1:
            idx=i
            break
    #print(idx,ans_w)
    return idx,ans_w



def pull2(a_t,feedback):
    w=feedback
    exam=[]
    for i in a_t:
        exam.append(w[i])
    idx=K-1
    ans_w=[]
    for i in range(K):
        ans_w.append(exam[i])
        if exam[i]==1:
            idx=i
            break
    return idx,ans_w

#obtain_x()
#pull_()