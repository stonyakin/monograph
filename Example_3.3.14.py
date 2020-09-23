import math
import random
import numpy as np
from datetime import datetime

def norm(x):
	if(np.linalg.norm(x)>R):
		x=x/np.linalg.norm(x)*R
	return x

def f(x):
	return sum(np.linalg.norm(np.subtract(x,i)) for i in Dots)

def grad_f(x):
	return sum(np.subtract(x,i)/np.linalg.norm(np.subtract(x,i)) for i in Dots)

def gen():
	t=[random.randint(-100,100) for i in range(n)]
	return t/np.linalg.norm(t)*(R-0.01)

n=int(1e5)
# N=[10,20,30,40,50,100,200,300,400,500,600,700,800,900,1000]
N=[10,20,30,40,50,100,200,300]
R=10
Rs=R**2/2
r=50 # Number of dots.
D=2 # Delta
x0=norm([0]*n)
alpha=0 # alpha_0
L0=1/math.sqrt(2)
l=len(N)
me=[[0,0,0,0] for i in range(l)]
mt=[[0,0,0,0] for i in range(l)]
m=10 # Number of experiments.

for j in range(m):
	print('Experiment #',j+1,'\r\n')
	Dots=[gen() for t in range(r)]
	L=L0 # L_0
	L/=2 # L_1
	Delta=1/200 # Delta_0
	Delta/=2 # Delta_1
	A=alpha # A_0
	x=x0 # x_0
	u=x0 # u_0
	S=0
	k=0
	i=0

	# print('FGM')
	# print('========================================================================================')

	# print('Adaptive')
	# print('================')

	start_time=datetime.now()
	while(i!=l):
		alpha=max(np.roots([L,-1,-A]))
		# if(alpha.imag!=0):
			# # print('Error!')
			# exit()
		A1=A+alpha
		y=norm(np.divide((np.multiply(alpha,u)+np.multiply(A,x)),A1))
		gf=grad_f(y)
		u1=norm(np.subtract(u,np.multiply(alpha,gf)))
		x1=norm(np.divide((np.multiply(alpha,u1)+np.multiply(A,x)),A1))
		t=x1-y
		if(f(x1)<=f(y)+np.dot(gf,t)+L/2*sum(np.power(t,2))+Delta*np.linalg.norm(t)):
			S+=Delta*np.linalg.norm(t)*A1
			L/=2
			Delta/=2
			x=x1
			u=u1
			A=A1
			k+=1
		else:
			L*=2
			# Delta*=2
			if(Delta<D):Delta*=2
		if(k==N[i]):
			end_time=datetime.now()
			t=(end_time-start_time).seconds
			me[i][0]+=(Rs+S)/A
			mt[i][0]+=t
			# print('K =',k)
			# print('Estimate =',(Rs+S)/A)
			# print('Time:',t,'s')
			# print('----------------------------------')
			i+=1

	# print('Non-adaptive')
	# print('================')

	L=L0 # L_0
	L/=2 # L_1
	A=alpha # A_0
	x=x0 # x_0
	S=0
	SN=0
	k=0
	i=0

	start_time=datetime.now()
	while(i!=l):
		alpha=max(np.roots([L,-1,-A]))
		# if(alpha.imag!=0):
			# # print('Error!')
			# exit()
		A1=A+alpha
		y=norm(np.divide((np.multiply(alpha,u)+np.multiply(A,x)),A1))
		gf=grad_f(y)
		u1=norm(np.subtract(u,np.multiply(alpha,gf)))
		x1=norm(np.divide((np.multiply(alpha,u1)+np.multiply(A,x)),A1))
		t=x1-y
		if(f(x1)<=f(y)+np.dot(gf,t)+L/2*sum(np.power(t,2))+D*np.linalg.norm(t)):
			S+=D*np.linalg.norm(t)*A1
			L/=2
			x=x1
			u=u1
			A=A1
			k+=1
		else:
			L*=2
		if(k==N[i]):
			end_time=datetime.now()
			t=(end_time-start_time).seconds
			me[i][1]+=(Rs+S)/A
			mt[i][1]+=t
			# print('K =',k)
			# print('Estimate =',(Rs+S)/A)
			# print('Time:',t,'s')
			# print('----------------------------------')
			i+=1

	# print()
	# print('GM')
	# print('========================================================================================')

	L=1/math.sqrt(2) # L_0
	L/=2 # L_1
	Delta=1/200 # Delta_0
	Delta/=2 # Delta_1
	x=x0 # x_0
	S=0
	SN=0
	k=0
	i=0

	# print('Adaptive')
	# print('================')

	start_time=datetime.now()
	while(i!=l):
		gf=grad_f(x)
		x1=norm(np.subtract(x,np.multiply(1/L,gf)))
		t=x1-x
		if(f(x1)<=f(x)+np.dot(gf,t)+L/2*sum(np.power(t,2))+Delta*np.linalg.norm(t)):
			S+=Delta*np.linalg.norm(t)/L
			SN+=1/L
			L/=2
			Delta/=2
			x=x1
			k+=1
		else:
			L*=2
			# Delta*=2
			if(Delta<D):Delta*=2
		if(k==N[i]):
			end_time=datetime.now()
			t=(end_time-start_time).seconds
			me[i][2]+=(Rs+S)/SN
			mt[i][2]+=t
			# print('K =',k)
			# print('Estimate =',(Rs+S)/SN)
			# print('Time:',t,'s')
			# print('----------------------------------')
			i+=1

	# print('Non-adaptive')
	# print('================')

	L=1/math.sqrt(2) # L_0
	L/=2 # L_1
	x=x0 # x_0
	S=0
	SN=0
	k=0
	i=0

	start_time=datetime.now()
	while(i!=l):
		gf=grad_f(x)
		x1=norm(np.subtract(x,np.multiply(1/L,gf)))
		t=x1-x
		if(f(x1)<=f(x)+np.dot(gf,t)+L/2*sum(np.power(t,2))+D*np.linalg.norm(t)):
			S+=D*np.linalg.norm(t)/L
			SN+=1/L
			L/=2
			x=x1
			k+=1
		else:
			L*=2
		if(k==N[i]):
			end_time=datetime.now()
			t=(end_time-start_time).seconds
			me[i][3]+=(Rs+S)/SN
			mt[i][3]+=t
			# print('K =',k)
			# print('Estimate =',(Rs+S)/SN)
			# print('Time:',t,'s')
			# print('----------------------------------')
			i+=1

	for s in range(l):
		print('K =',N[s])
		print('------------------------------------------')
		print('FGM adaptive:')
		print('Mean estimate =',me[s][0]/(j+1))
		print('Mean time:',mt[s][0]/(j+1),'s')
		print('FGM non-adaptive:')
		print('Mean estimate =',me[s][1]/(j+1))
		print('Mean time:',mt[s][1]/(j+1),'s')
		print('GM adaptive:')
		print('Mean estimate =',me[s][2]/(j+1))
		print('Mean time:',mt[s][2]/(j+1),'s')
		print('GM non-adaptive:')
		print('Mean estimate =',me[s][3]/(j+1))
		print('Mean time:',mt[s][3]/(j+1),'s')
		print('------------------------------------------')
	print()
