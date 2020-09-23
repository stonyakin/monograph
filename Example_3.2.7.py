import math
import random
import numpy as np
from datetime import datetime

def norm(x):
	if(np.linalg.norm(x)>R):
		x=x/np.linalg.norm(x)*R
	return x

def f(x):
	return sum(np.power(np.dot(C,x)-b,2))/2-e

def grad_f(x):
	return np.dot(Ct,np.dot(C,x)-b)

def gen():
	t=[random.randint(-100,100) for i in range(n)]
	return t/np.linalg.norm(t)*(R-0.01)

n=int(1e3)
R=10
x0=norm([0]*n)
N=[10,20,30,40,50,100,200,300,400,500,600,700,800,900,1000]
Rs=R**2/2
d=1/20 # delta
D=1/20 # Delta
ln=len(N)
me=[[0,0] for i in range(ln)]
mf=[[0,0] for i in range(ln)]
mt=[[0,0] for i in range(ln)]
ex=5 # number of experiments
q=1

while(q<=ex):
	print('Experiment #',q,'\n')
	C=np.diagflat([random.randint(1,1000) for i in range(n)])
	for i in range(100):C[random.randint(0,n-1)][random.randint(0,n-1)]=random.randint(1,1000)
	Ct=np.transpose(C)
	b=[random.randint(-1000,1000) for i in range(n)]
	L_0=max(np.linalg.eig(np.dot(Ct,C))[0]).real
	e=random.uniform(0,d) # delta_hat
	L=L_0 # L_0
	L/=2 # L_1
	delta=1/200 # delta_0
	delta/=2 # delta_1
	Delta=1/200 # Delta_0
	Delta/=2 # Delta_1
	alpha=0 # alpha_0
	A=alpha # A_0
	x=x0 # x_0
	u=x0 # u_0
	S=0
	k=0
	i=0
	j=0
	F=None
	print('Adaptive')
	print('================')
	start_time=datetime.now()
	while(i!=len(N)):
		fx=f(x)
		if(F==None or fx<F):
			F=fx
		alpha=max(np.roots([L,-1,-A]))
		A1=A+alpha
		y=norm(np.divide((np.multiply(alpha,u)+np.multiply(A,x)),A1))
		gf=grad_f(y)
		u1=norm(np.subtract(u,np.multiply(alpha,gf)))
		x1=norm(np.divide((np.multiply(alpha,u1)+np.multiply(A,x)),A1))
		t=x1-y
		if(f(x1)<=f(y)+np.dot(gf,t)+L/2*sum(np.power(t,2))+Delta*np.linalg.norm(t)+delta):
			S+=A1*(Delta*np.linalg.norm(t)+delta+d)
			L/=2
			delta/=2
			Delta/=2
			x=x1
			u=u1
			A=A1
			k+=1
		else:
			L*=2
			delta*=2
			Delta*=2
		if(k==N[i]):
			print('Iterations:',k)
			te=(Rs+S)/A
			me[j][0]+=te
			print('Estimate:',round(te,4))
			mf[j][0]+=F
			print('F(x):',F)
			end_time=datetime.now()
			t=(end_time-start_time).seconds
			mt[j][0]+=t
			print('Time:',t,'s')
			print('----------------------------------')
			i+=1
			j+=1
	print('Non-adaptive')
	print('================')
	L=L_0 # L_0
	L/=2 # L_1
	alpha=0 # alpha_0
	A=alpha # A_0
	x=x0 # x_0
	S=0
	SN=0
	k=0
	i=0
	j=0
	F=None
	start_time=datetime.now()
	while(i!=len(N)):
		fx=f(x)
		if(F==None or fx<F):
			F=fx
		alpha=max(np.roots([L,-1,-A]))
		A1=A+alpha
		y=norm(np.divide((np.multiply(alpha,u)+np.multiply(A,x)),A1))
		gf=grad_f(y)
		u1=norm(np.subtract(u,np.multiply(alpha,gf)))
		x1=norm(np.divide((np.multiply(alpha,u1)+np.multiply(A,x)),A1))
		t=x1-y
		if(f(x1)<=f(y)+np.dot(gf,t)+L/2*sum(np.power(t,2))+D*np.linalg.norm(t)+d):
			S+=A1*(D*np.linalg.norm(t)+2*d)
			L/=2
			x=x1
			u=u1
			A=A1
			k+=1
		else:
			L*=2
		if(k==N[i]):
			print('Iterations:',k)
			te=(Rs+S)/A
			me[j][1]+=te
			print('Estimate:',round(te,4))
			mf[j][1]+=F
			print('F(x):',F)
			end_time=datetime.now()
			t=(end_time-start_time).seconds
			mt[j][1]+=t
			print('Time:',t,'s')
			print('----------------------------------')
			i+=1
			j+=1
	print()
	for i in range(ln):
		print('Iterations:',N[i])
		print('Mean estimate (adaptive):',round(me[i][0]/q,5))
		print('Mean F(x) (adaptive):',round(mf[i][0]/q,5))
		print('Mean time (adaptive):',mt[i][0]/q,'s')
		print('Mean estimate (non-adaptive):',round(me[i][1]/q,5))
		print('Mean F(x) (non-adaptive):',round(mf[i][1]/q,5))
		print('Mean time (non-adaptive):',mt[i][1]/q,'s')
	q+=1
	print()
