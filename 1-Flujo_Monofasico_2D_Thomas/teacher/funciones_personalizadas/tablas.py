import numpy as np
import pandas as pd

def color_zeros(valor):
    color = 'green' if valor == 0 else 'dark'
    return 'color: %s' % color

def generar_matriz(nodos_x, nodos_y, nodo_a_discretizar):
    n=nodos_x*nodos_y
    matriz = np.zeros((n,n), dtype='object') # es de tipo objeto porque se combinan numeros y letras

    for j in range (0,nodos_y):
        for i in range (0,nodos_x):
            k=i+nodos_x*j     
            matriz[k][k]='AP'
            if j < nodos_y-1: matriz[k][k+nodos_x]='AN'
            if i < nodos_x-1: matriz[k][k+1]='AE'
            if i > 0: matriz[k][k-1]='AW'
            if j > 0: matriz[k][k-nodos_x]='AS'
    
    df = pd.DataFrame(matriz)
    idx = pd.IndexSlice
    slice_ =  idx[idx[nodo_a_discretizar, :]]
    s = df.style.applymap(color_zeros)
    s.applymap(color_zeros, subset=slice_).set_properties(**{'background-color': 'lightblue'}, subset=slice_)
    return(s)