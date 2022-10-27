import numpy as np
import matplotlib.pyplot as plt

def crea_malla(lx, ly, nx, ny):
    xg, yg = np.meshgrid(np.linspace(0,lx,nx), np.linspace(0,ly,ny))
    return xg, yg

def discretizacion_en_malla_rectangular(lx, ly, nodos_x, nodos_y, nx_a_discretizar = None, ny_a_discretizar = None):
    # hx y hy son los espaciamientos entre nodos |--hx--|--hx--|
    hx = lx/nodos_x-1
    hy = ly/nodos_y-1
    
    # generamos la malla (arreglo en 2D)
    malla_x_2d, malla_y_2d = np.meshgrid(np.linspace(0,lx,nodos_x), np.linspace(0,ly,nodos_y))
    
    # generamos un arreglo 1D para visualizar la numeración de nodos
    malla_x_1d = malla_x_2d.flatten() 
    malla_y_1d = malla_y_2d.flatten()
    
    # graficar la malla
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
    
    # primer grafico
    ax1.set_title("Numeración de Nodos")
    ax1.plot(malla_x_2d, malla_y_2d, 'y-', malla_x_2d.transpose(), malla_y_2d.transpose(), 'y-')
    ax1.plot(malla_x_1d, malla_y_1d, 'co')
    # se enumeran los nodos de la malla
    for i in range(len(malla_x_1d)):
        ax1.annotate(str(i), (malla_x_1d[i]+hx/10, malla_y_1d[i]+hy/10), color='black')    
    ii, jj = nx_a_discretizar, ny_a_discretizar
    
    if nx_a_discretizar != None:
        ax1.annotate("AP", (malla_x_2d[jj][ii]+hx/10, malla_y_2d[jj][ii]-hy/10), fontsize=12, color='red')
        if jj+1 < nodos_y: ax1.annotate("AN", (malla_x_2d[jj][ii]+hx/10, malla_y_2d[jj+1][ii]-hy/10), fontsize=12, color='blue')
        if ii+1 < nodos_x: ax1.annotate("AE", (malla_x_2d[jj][ii+1]+hx/10, malla_y_2d[jj][ii]-hy/10), fontsize=12, color='blue')
        if ii > 0: ax1.annotate("AW", (malla_x_2d[jj-1][ii-1]+hx/10, malla_y_2d[jj][ii]-hy/10), fontsize=12, color='blue')
        if jj > 0: ax1.annotate("AS", (malla_x_2d[jj][ii]+hx/10, malla_y_2d[jj-1][ii-1]-hy/10), fontsize=12, color='blue')
    
###    ax1.tick_params(axis='x', colors='white')
###    ax1.tick_params(axis='y', colors='white')
###    ax1.spines['top'].set_color('white')
    #ax1.spines['left'].set_color('white')
###    ax1.spines['right'].set_color('white')
    #ax1.spines['bottom'].set_color('white')
    
### Decoración de los ejes
    ax1.axes.xaxis.set_ticklabels([])
    ax1.axes.yaxis.set_ticklabels([])
    ax1.set_xlabel('Lx')
    ax1.set_ylabel('Ly')
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.axes.xaxis.set_ticks([])
    ax1.axes.yaxis.set_ticks([])
    xoffset = lx * 0.05
    yoffset = ly * 0.05
    ax1.annotate('', xy=(0,-yoffset), xytext=(lx,-yoffset), arrowprops=dict(arrowstyle='<->'))
    ax1.annotate('', xy=(-xoffset,0), xytext=(-xoffset,ly), arrowprops=dict(arrowstyle='<->'))
            
    arr_shape = np.shape(malla_x_2d) 
    nx, ny = arr_shape[0], arr_shape[1]
    
    # segundo grafico
    ax2.set_title("Índices de Nodos")
    ax2.plot(malla_x_2d, malla_y_2d, 'y-', malla_x_2d.transpose(), malla_y_2d.transpose(), 'y-')
    ax2.plot(malla_x_1d, malla_y_1d, 'co')
    
    for j in range(0, ny):
        for i in range(0, nx):
            ax2.annotate(f"({j}, {i})", (malla_x_2d[i][j]+hx/10, malla_y_2d[i][j]+hy/10), fontsize=8, color='black')
#    ii, jj = nx_a_discretizar, ny_a_discretizar

    if ny_a_discretizar != None:
        ax2.annotate("AP", (malla_x_2d[jj][ii]+hx/10, malla_y_2d[jj][ii]-hy/10), fontsize=12, color='red')
        if jj+1 < nodos_y: ax2.annotate("AN", (malla_x_2d[jj][ii]+hx/10, malla_y_2d[jj+1][ii]-hy/10), fontsize=12, color='blue')
        if ii+1 < nodos_x: ax2.annotate("AE", (malla_x_2d[jj][ii+1]+hx/10, malla_y_2d[jj][ii]-hy/10), fontsize=12, color='blue')
        if ii > 0: ax2.annotate("AW", (malla_x_2d[jj-1][ii-1]+hx/10, malla_y_2d[jj][ii]-hy/10), fontsize=12, color='blue')
        if jj > 0: ax2.annotate("AS", (malla_x_2d[jj][ii]+hx/10, malla_y_2d[jj-1][ii-1]-hy/10), fontsize=12, color='blue')

### Decoración de los ejes
    ax2.axes.xaxis.set_ticklabels([])
    ax2.axes.yaxis.set_ticklabels([])
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.axes.xaxis.set_ticks([])
    ax2.axes.yaxis.set_ticks([])
    ax2.annotate('', xy=(malla_x_2d[0][0],-yoffset), xytext=(malla_x_2d[0][1],-yoffset), arrowprops=dict(arrowstyle='<->'))
    ax2.annotate('', xy=(-xoffset, malla_y_2d[0][0]), xytext=(-xoffset,malla_y_2d[1][0]), arrowprops=dict(arrowstyle='<->'))
    ax2.text(x = malla_x_2d[0][0]+hx/2, y = -2*yoffset, s='hx')
    ax2.text(x = -2*xoffset, y = malla_y_2d[0][0]+hy/2, s='hy')

    """
    ax2.axes.xaxis.set_ticklabels([])
    ax2.axes.yaxis.set_ticklabels([])
    
    ax2.set_xlabel('   hx  --|--  hx  --|--   hx   ')
    ax2.set_ylabel('   hy  --|--  hy  --|--   hy   ')
    
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    
    ax2.spines['top'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['right'].set_color('white')
    ax2.spines['bottom'].set_color('white')
    """

    plt.show()
    if nx_a_discretizar != None and ny_a_discretizar != None:
        print("Nodo discretizado: ({},{})".format(ii,jj))
#    return (ii, jj)

def graficar_isobaras_presion(malla_x, malla_y, Press):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_title("Isobaras (FRONTERA NORTE)")
    ax.contourf(malla_x, malla_y, Press.transpose(), 10, alpha=.85)
    C = ax.contour(malla_x, malla_y, Press.transpose(),10, colors='black', linewidths =1.0)
    ax.clabel(C,inline=1,fontsize=10)
    ax.set_xlabel('$ x(ft) $ (FRONTERA SUR)', fontsize=12)
    ax.set_ylabel('$ y(ft)$ (FRONTERA ESTE)', fontsize=12)
    ax.text(malla_x[-1][-1], 250, '(FRONTERA OESTE)', fontsize=12, horizontalalignment='left', verticalalignment='center', rotation=90, clip_on=False )
    plt.grid()
    plt.show()


def graficar_isobaras_presion_y_campo_velocidad(malla_x, malla_y, Press, u, v):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 8))
    
    #Grafico 1 presion
    ax1.set_title("Isobaras (FRONTERA NORTE)")
    ax1.contourf(malla_x, malla_y, Press.transpose(), 10, alpha=.85)
    C = ax1.contour(malla_x, malla_y, Press.transpose(),10, colors='black', linewidths =1.0)
    ax1.clabel(C,inline=1,fontsize=10)
    ax1.set_xlabel('$ x(ft) $ (FRONTERA SUR)', fontsize=12)
    ax1.set_ylabel('$ y(ft)$ (FRONTERA ESTE)', fontsize=12)
    ax1.text(525, 250, '(FRONTERA OESTE)', fontsize=12, horizontalalignment='left', verticalalignment='center', rotation=90, clip_on=False )
    plt.grid()

    '''
    # Grafico 2 campo de velocidades
    u = np.zeros((nx,ny))
    v = np.zeros((nx,ny))
    for j in range (1,ny-1):
        for i in range (1,nx-1):
            u[i][j]=-(Press[i+1][j]-Press[i-1][j])/(2*hx)  #Discretizamos por DFC de primer orden
            v[i][j]=-(Press[i][j+1]-Press[i][j-1])/(2*hy)
    '''
    plt.title("Campo vectorial de velocidad (FRONTERA NORTE)", fontsize=12)
    ax2.quiver(malla_x, malla_y, u.transpose(), v.transpose(), color='b', scale=10)
    ax2.set_xlabel('$ x(ft) $ (FRONTERA SUR)', fontsize=12)
    ax2.set_ylabel('$ y(ft)$ (FRONTERA ESTE)', fontsize=12)
    ax2.text(525, 250, '(FRONTERA OESTE)', fontsize=12, horizontalalignment='left', verticalalignment='center', rotation=90, clip_on=False )
    plt.grid()
    
    plt.show()
    