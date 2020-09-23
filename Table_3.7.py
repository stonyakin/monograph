import math
import random
import numpy as np
from datetime import datetime

def norm(x):
	return sum(abs(x))

def g(x):
	u=x[:n]
	v=x[n:l]
	return np.append(np.matmul(At,v)+random.uniform(-D,D)/(2*n),np.matmul(-A,u)+random.uniform(-D,D)/(2*m))

def h(x):
	u=x[:n]
	v=x[n:l]
	tu=u*np.exp(-np.matmul(At,v)/L)
	tv=v*np.exp(-np.matmul(-A,u)/L)
	return np.append(tu/sum(tu),tv/sum(tv))

def V(y,x):
	for i in range(l):
		if(x[i]==0):x[i]=1e-5
		if(y[i]==0):y[i]=1e-5
	return sum(y*np.log(y/x))

n=1000
m=2000
l=n+m
N=[10,20,50,100,200,300,400,500,1000]
D=1/12
Delta=1/100
Rs=math.log(n)+math.log(m)
ln=len(N)
ex=5 # Number of experiments
mt1=[0 for i in range(ln)]
mt2=[0 for i in range(ln)]
me1=[0 for i in range(ln)]
me2=[0 for i in range(ln)]

for j in range(ex):
	A=np.random.randint(10,size=(m,n))
	At=np.transpose(A)
	L=math.sqrt(2)/2 # L_0
	L/=2 # L_1
	x_1=[1/n]*n+[1/m]*m # x_0
	x_2=[1/n]*n+[1/m]*m # x_0
	SN=SY=S1=S2=k1=k2=t1=t2=0
	print('Experiment #',j+1,'\r\n')
	for i in range(ln):
		print('N =',N[i])

		start_time=datetime.now()
		while(k1<N[i]):
			y=h(x_1)
			x1=h(y)
			if(np.dot(g(y)-g(x_1),y-x1)<=L*(V(y,x_1)+V(y,x1))+D*norm(y-x1)):
				SN+=1/L
				SY+=y/L
				S1+=1/L
				L/=2
				x_1=x1
				k1+=1
			else:
				L*=2
		# Y=SY/SN
		# print('Output result:',Y)
		end_time=datetime.now()
		t1+=(end_time-start_time).seconds
		mt1[i]+=t1
		me1t=1/SN*(Rs)+8*D
		me1[i]+=me1t
		print('Estimate (non-adaptive) =',round(me1t,5))
		print('Time (non-adaptive):',t1,'s')

		start_time=datetime.now()
		while(k2<N[i]):
			y=h(x_2)
			x1=h(y)
			if(np.dot(g(y)-g(x_2),y-x1)<=L*(V(y,x_2)+V(y,x1))+Delta*norm(y-x1)):
				SN+=1/L
				SY+=y/L
				S2+=Delta*np.linalg.norm(y-x1)/L
				L/=2
				Delta/=2
				x_2=x1
				k2+=1
			else:
				L*=2
				Delta*=2
		# Y=SY/SN
		# print('Output result:',Y)
		end_time=datetime.now()
		t2+=(end_time-start_time).seconds
		mt2[i]+=t2
		me2t=1/SN*(Rs+S2)+4*D
		me2[i]+=me2t
		print('Estimate (adaptive) =',round(me2t,5))
		print('Time (adaptive):',t2,'s')

	print()
	for i in range(ln):
		print('Iterations =',N[i])
		print('Mean estimate (non-adaptive) =',round(me1[i]/(j+1),5))
		print('Mean time (non-adaptive):',mt1[i]/(j+1),'s')
		print('Mean estimate (adaptive) =',round(me2[i]/(j+1),5))
		print('Mean time (adaptive):',mt2[i]/(j+1),'s')
	print('\r\n-----------------------\r\n')
