# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:54:21 2022

@author: leo_teja
"""


import matplotlib.pyplot as plt     #importa la biblioteca matplotlib con el sobrenombre plt (gráficar) 
import solver as SL              #importa el archivo propio solverLin con el sobrenombre SL (resolver sistemas tridiagonales) 
import numpy as np                  #importa la biblioteca numpy para utilizar matrices y vectores (numerical python)
from time import time

"""
Este código resuelve la siguiente ecuación diferencial parcial, de segundo orden
lineal mediante diferencias finitas (técnica de discretización):

    (Mu*Phi*Ct)/k dP/dt =d**2P/dr**2+(1/r)*dP/dr
    
    
 P=Pa |o---o---o----o----o----o----o----o---o----o---o| P=Pb
      rw                                              re
                          Lr

C1=(Mu*Phi*Ct)/k

  cP*(1/psi)/mD*(psi/dia)=psi/ft**2+psi/ft**2

cP --> psi*dia
mD --> ft**2


"""

Nr=100    #Numero de nodos del espacio discreto

# Pa=5000; Pb=6000  #psi 
# rw=5/12; re=3000.0   #ft
# Pinicial=Pb
# tiempoSimulacion=1 #días
# deltat=0.1  #días
# q=-1000  #rb/day

#tipo 1  #Pa y Pb  psi  a=rw #radio del pozo en ft          
Pa=2000;  Pb=3000;  rw=0.5; re=500.5      

tiempoSimulacion=2000 #Días
deltat=1.0  #Días

k=100.0    #Permeabilidad en mD
phi=0.1   #Porosidad 
Cr=5e-07 #1/psi
Cf=2e-06  #1/psi
Mu=50.0  #cP
Pinicial=Pa   #Presión inicial en psi
q=-500.0 #rb/day
A=1000  # ft2


r=np.linspace(rw,re,Nr)  #genera la variable independiente

NrInt=Nr-2   #Nodos internos donde las incognitas "p" son desconocidas

Lr=re-rw
deltar=Lr/(Nr-1) 

#Propiedades del fluido y de la roca

Ct=Cf+Cr

k=k*1.0623e-14   #mD -------> ft**2
Mu=Mu*1.45e-07   #cP------> psi*s
deltat=deltat*86400.0  #dia ----> s
q=q*5.61        #rb/day ---ft**3/day
q=q/86400.0     #ft**3/day ---> ft**3/s 

C1=(Mu*phi*Ct)/k
C2=(q*Mu*deltar)/(A*k)


AP=np.zeros(NrInt)
AE=np.zeros(NrInt)
AW=np.zeros(NrInt)
B=np.zeros(NrInt)
Pi=np.zeros(NrInt)
Press=np.zeros(Nr)


P_old=np.ones(NrInt)*Pinicial

valorMaxP=10.0
tolP=1.0

#Solución por el método implicito y con derivadas centrales en el espacio

tiempo=0.0
while ((tiempo<tiempoSimulacion)and(valorMaxP>tolP)):
    
    for i in range (0,NrInt):
        
        AP[i]=-(2/deltar**2+C1/deltat)
        AE[i]= 1/deltar**2+1/(r[i+1]*2*deltar)
        AW[i]= 1/deltar**2-1/(r[i+1]*2*deltar)
        B[i]=-C1*P_old[i]/deltat
        
    #Condición de frontera en rw (Pa conocida)
    B[0]=B[0]-AW[0]*Pa

    #Condición de frontera en rw (Q constate)
    # B[0]=B[0]-AW[0]*C2
    # AP[0]=AP[0]+AW[0]

    
    #Condición de frontera en re (Pb conocida)    
    B[NrInt-1]=B[NrInt-1]-AE[NrInt-1]*Pb
        
    #Solución mediante el algortimo de thomas
    Pi=SL.thomas1D(AP, -AE, -AW, B)   #
    
    # Pa=C2+Pi[0]  #Presión calculada cuando usamos una condición de frontera a q=cte

    valorMaxP=np.max(abs(P_old-Pi))  #Paro estado estable
    
    #Actualización de la presión anterior para el siguiente paso de tiempo
    P_old=np.copy(Pi)
    
    tiempo=tiempo+deltat/86400.0
    
    Press[0]=Pa
    Press[Nr-1]=Pb
    Press[1:Nr-1]=Pi

    
    plt.figure("solucion MI")
    plt.plot(r, Press)
    
    print("tiempo de simulación:", tiempo, "\t", "diferencia de P:",  valorMaxP)
    
plt.grid()


#SOLUCIÓN SEMI-IMPLICITO CRANK-NICHOLSON

AP=np.zeros(NrInt)
AE=np.zeros(NrInt)
AW=np.zeros(NrInt)
B=np.zeros(NrInt)
Pie=np.zeros(NrInt)
Press=np.zeros(Nr)


P_old=np.ones(Nr)*Pinicial
P_old[0]=Pa
P_old[Nr-1]=Pb


valorMaxP=10.0
tolP=1
deltat=deltat/200

#Solución por el método implicito y con derivadas centrales en el espacio

tiempo=0.0
while ((tiempo<tiempoSimulacion)and(valorMaxP>tolP)):
    
    for i in range (1,Nr-1):
        GEx=(P_old[i-1]-2*P_old[i]+P_old[i+1])/deltar**2+(1/r[i])*((P_old[i+1]-P_old[i-1])/(2*deltar))
        
        AP[i-1]=-(2/deltar**2+2*C1/deltat)
        AE[i-1]= 1/deltar**2+1/(r[i]*2*deltar)
        AW[i-1]= 1/deltar**2-1/(r[i]*2*deltar)
        B[i-1]=-2*C1*P_old[i]/deltat-GEx

        
    #Condición de frontera en rw (Pa conocida)
    B[0]=B[0]-AW[0]*Pa

    #Condición de frontera en rw (Q constate)
    # B[0]=B[0]-AW[0]*C2
    # AP[0]=AP[0]+AW[0]

    
    #Condición de frontera en re (Pb conocida)    
    B[NrInt-1]=B[NrInt-1]-AE[NrInt-1]*Pb
        
    #Solución mediante el algortimo de thomas
    Pie=SL.thomas1D(AP, -AE, -AW, B)   #
    
    # Pa=C2+Pi[0]  #Presión calculada cuando usamos una condición de frontera a q=cte

    valorMaxP=np.max(abs(P_old[1:Nr-1]-Pie))  #Paro estado estable
    
    #Actualización de la presión anterior para el siguiente paso de tiempo
    P_old[1:Nr-1]=np.copy(Pie)
    
    tiempo=tiempo+deltat/86400.0
    
    Press[0]=Pa
    Press[Nr-1]=Pb
    Press[1:Nr-1]=Pie

    
    plt.figure("solucion MIe")
    plt.plot(r, Press)
    
    print("tiempo de simulación:", tiempo, "\t", "diferencia de P:",  valorMaxP)
    
plt.grid()
plt.show()




"""
AP=np.zeros(NrInt)
AE=np.zeros(NrInt)
AW=np.zeros(NrInt)
B=np.zeros(NrInt)

Pe=np.zeros(NrInt)
P_old=np.ones(NrInt)*Pinicial


#tiempoSimulacion=1
#Solución por el método explicíto y con derivadas centrales en el espacio
deltat=deltat/1000

tiempo=0.0
while (tiempo<tiempoSimulacion):
    
    for i in range (0,NrInt):
        
        AP[i]=-2/deltar**2
        AE[i]= 1/deltar**2+1/(r[i+1]*2*deltar)
        AW[i]= 1/deltar**2-1/(r[i+1]*2*deltar)
        B[i]=P_old[i]
        
        if (i>0 and i<NrInt-1):
            Pe[i]=deltat/C1*(AW[i]*P_old[i-1]+AP[i]*P_old[i]+AE[i]*P_old[i+1])+B[i]
            
        #Condición de frontera en rw (Pa conocida)
        if (i==0):
            Pe[i]=deltat/C1*(AW[i]*Pa+AP[i]*P_old[i]+AE[i]*P_old[i+1])+B[i]
 
        #Condición de frontera en re (Pb conocida)    
        if (i==NrInt-1):
            Pe[i]=deltat/C1*(AW[i]*P_old[i-1]+AP[i]*P_old[i]+AE[i]*Pb)+B[i]
            
    # #Condición de frontera en rw (Pa conocida)
    # AP[0]=-2/deltar**2    
    # AE[0]=1/deltar**2+1/(r[1]*2*deltar)
    # AW[0]=1/deltar**2-1/(r[1]*2*deltar)
    # B[0]=P_old[0]
    
    # Pe[0]=deltat/C1*(AW[0]*Pa+AP[0]*P_old[0]+AE[0]*P_old[1])+B[0]
    
    # # #Condición de frontera en re (Pb conocida)    
    # AP[NrInt-1]=-2/deltar**2    
    # AE[NrInt-1]=1/deltar**2+1/(r[NrInt]*2*deltar)
    # AW[NrInt-1]=1/deltar**2-1/(r[NrInt]*2*deltar)
    # B[NrInt-1]=P_old[NrInt-1]
        
    # Pe[NrInt-1]=deltat/C1*(AW[NrInt-1]*P_old[NrInt-2]+AP[NrInt-1]*P_old[NrInt-1]+AE[NrInt-1]*Pb)+B[NrInt-1]

    # #Actualización de la presión anterior para el siguiente paso de tiempo
    P_old=np.copy(Pe)
    
    tiempo=tiempo+deltat/86400
    
    plt.figure("solucion ME")
    plt.plot(r[1:Nr-1], Pe)
    
    print("tiempo de simulación:", tiempo)
    
#plt.grid()

"""

