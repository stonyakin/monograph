import numpy as np
import math
from datetime import datetime
import random

def Norm(x):
	if(np.linalg.norm(x)>R):
		x=R*x/np.linalg.norm(x)
	# for i in range(n):
		# if(x[i]<=0):x[i]=1e-10
	return np.array(x)

def G(g,b,x,tr):
	if(tr=='grad'):
		if(tg=='linear'):t=g
		if(tg=='abs'):t=g*np.sign(x)
		if(tg=='sqr'):t=g*np.multiply(2,x)
	if(tr=='g'):
		if(tg=='linear'):t=sum(g*x)-b
		if(tg=='abs'):t=sum(g*abs(x))-b
		if(tg=='sqr'):t=sum(g*x**2)-b
	return t

# Modification.
def Grad_G_M(x):
	global M,v
	for i in range(len(C)):
		M=np.linalg.norm(C[i])
		v=G(C[i],b[i],x,'g')
		if(v/M>e[j]):
			return G(C[i],None,x,'grad')
	return [None]

# Standard.
def Grad_G_S(x):
	global M,v
	v=G(C[0],b[0],x,'g')
	gm=C[0]
	for i in range(len(C)):
		t=G(C[i],b[i],x,'g')
		if(t>v):
			v=t
			gm=C[i]
	M=np.linalg.norm(gm)
	if(v/M>e[j]):
		return G(gm,None,x,'grad')
	return [None]

def F(x):
	return -np.dot(f,x)

def Grad_F(x):
	return np.multiply(-1,f)

n=20
m=200
tg='linear'
R=5
x0=Norm([1e-3]*n)
Th0s=R**2/2
e=[1/2,1/3,1/4,1/5,1/6]
E=['1/2','1/3','1/4','1/5','1/6']
ln=len(e)
mi=[[0,0] for i in range(ln)]
mt=[[0,0] for i in range(ln)]
ex=5 # Number of experiments
q=1

while(q<=ex):
	C=np.random.randint(-10,11,size=(m,n))
	for i in C:
		if(sum(i)==0):
			i[random.randint(0,n-1)]=1
	b=np.random.randint(1,6,size=m)
	# b=[-1]*m
	f=np.random.randint(1,11,size=n)
	print('Experiment #',q,'\n')
	print('Algorithm N')
	print('====================================')
	for j in range(ln):
		print('Epsilon:',E[j])
		x=x0
		k=0
		I=0
		S=0
		start_time=datetime.now()
		while(2*Th0s/(e[j]/n)**2>k):
			grad_g=Grad_G_S(x)
			if(grad_g[0]==None):
				grad_f=Grad_F(x)
				M=np.linalg.norm(grad_f)
				h=e[j]/M/n
				y=Norm(np.subtract(x,np.multiply(h,grad_f)))
				if(I==0 or F(X)>F(x)):
					X=x
				I+=1
				S+=1/M
			else:
				h=e[j]/M/n
				y=Norm(np.subtract(x,np.multiply(h,grad_g)))
			x=y
			k+=1
		if(I!=0):
			mi[j][0]+=k
			print('Iterations:',k)
			# print('Ensure:',X)
		else:
			print("The set I is empty, k =",k)
		end_time=datetime.now()
		t=(end_time-start_time).seconds
		mt[j][0]+=t
		print('Time:',t,'s')
		print('------------------------------------')
	print()
	print('Algorithm N-2')
	print('====================================')
	for j in range(ln):
		print('Epsilon:',E[j])
		x=x0
		k=0
		I=0
		S=0
		start_time=datetime.now()
		while(2*Th0s/(e[j]/n)**2>k):
			grad_g=Grad_G_M(x)
			if(grad_g[0]==None):
				grad_f=Grad_F(x)
				M=np.linalg.norm(grad_f)
				h=e[j]/M/n
				y=Norm(np.subtract(x,np.multiply(h,grad_f)))
				if(I==0 or F(X)>F(x)):
					X=x
				I+=1
				S+=1/M
			else:
				h=e[j]/M/n
				y=Norm(np.subtract(x,np.multiply(h,grad_g)))
			x=y
			k+=1
		if(I!=0):
			mi[j][0]+=k
			print('Iterations:',k)
			# print('Ensure:',X)
		else:
			print("The set I is empty, k =",k)
		end_time=datetime.now()
		t=(end_time-start_time).seconds
		mt[j][0]+=t
		print('Time:',t,'s')
		print('------------------------------------')
	print()
	print('Algorithm J')
	print('====================================')
	for j in range(ln):
		print('Epsilon:',E[j])
		x=x0
		k=0
		I=0
		S=0
		Sh=0
		Shx=0
		start_time=datetime.now()
		while(2*Th0s/e[j]**2>S+k-I):
			grad_g=Grad_G_S(x)
			if(grad_g[0]==None):
				grad_f=Grad_F(x)
				Ms=sum(np.power(grad_f,2)) # Square norm.
				h=e[j]/Ms
				y=Norm(np.subtract(x,np.multiply(h,grad_f)))
				I+=1
				S+=1/Ms
				Sh+=h
				Shx+=h*x
			else:
				h=e[j]/M
				y=Norm(np.subtract(x,np.multiply(h,grad_g)))
			x=y
			k+=1
		if(I!=0):
			X=Shx/Sh
			mi[j][1]+=k
			print('Iterations:',k)
			# print('Ensure:',X)
		else:
			print("The set I is empty, k =",k)
		end_time=datetime.now()
		t=(end_time-start_time).seconds
		mt[j][1]+=t
		print('Time:',t,'s')
		print('------------------------------------')
	print()
	for i in range(ln):
		print('Epsilon:',E[i])
		print('Mean iterations (N):',round(mi[i][0]/q,5))
		print('Mean time (N):',mt[i][0]/q,'s')
		print('Mean iterations (I):',round(mi[i][1]/q,5))
		print('Mean time (I):',mt[i][1]/q,'s')
	q+=1
	print()
