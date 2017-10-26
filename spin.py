import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math
import scipy.sparse.linalg as sla
import time

#defining the Hilbert space, Hamiltonian parameters, temperature, etc
start0 = time.time()
size = 10
dim = 2**size
energy = np.empty((dim,dim),'complex')
J = 1
T = 4

#2by2 pauli matrix
sigx = [[0,1],[1,0]]
sigy = [[0,-1j],[1j,0]]
sigz = [[1,0],[0,-1]]

print 'The ',size,' spin chain with dimension ',dim,' Hilbert space.'

#getting states from an integer, return a 2 by n matrix represent a state such as |00...010>
def states(n):
    counter = n
    t = 0
    f = np.zeros((2,size))
    for i in range(0,size):
        f[1][i]=1
    if counter > 512.5:
        f[0][t]=1
        f[1][t]=0
        counter = counter - 512
    t = t+1
    if counter > 256.5:
        f[0][t]=1
        f[1][t]=0
        counter = counter - 256
    t = t+1
    if counter > 128.5:
        f[0][t]=1
        f[1][t]=0
        counter = counter - 128
    t = t+1
    if counter > 64.5:
        f[0][t]=1
        f[1][t]=0
        counter = counter - 64
    t = t+1
    if counter > 32.5:
        f[0][t]=1
        f[1][t]=0
        counter = counter - 32
    t = t+1
    if counter > 16.5:
        f[0][t]=1
        f[1][t]=0
        counter = counter - 16
    t = t+1
    if counter > 8.5:
        f[0][t]=1
        f[1][t]=0
        counter = counter - 8
    t = t+1
    if counter > 4.5:
        f[0][t]=1
        f[1][t]=0
        counter = counter - 4
    t = t+1
    if counter > 2.5:
        f[0][t]=1
        f[1][t]=0
        counter = counter - 2
    t = t+1
    if counter > 1.5:
        f[0][t]=1
        f[1][t]=0
    t = t+1
            
    return f

#Calculate the Hamiltonian matrix element

def matelt(a,b):
    element = 0
    brav = np.empty((size,2))
    ketv = np.empty((size,2))


    bra = states(a)
    ket = states(b)
    diff = bra - ket
    if LA.norm(diff[0]) > 1.7:
        return 0
    #print 'The bra corresponds to',a,'is',bra
    #print 'The ket corresponds to',b,'is',ket
    for i in range(0,size):
        
        brav[i][0] = bra[0][i]
        brav[i][1] = bra[1][i]

        ketv[i][0] = ket[0][i]
        ketv[i][1] = ket[1][i]

 
        # can use different Hamiltonian here
        #element = element + (-0.5)*(bravT1.dot(sigx).dot(ketv1))*(bravT2.dot(sigx).dot(ketv2)) + (-0.5)*(bravT1.dot(sigy).dot(ketv1))*(bravT2.dot(sigy).dot(ketv2)) + (-0.5)*(bravT1.dot(sigz).dot(ketv1))*(bravT2.dot(sigz).dot(ketv2))
    #print brav
    #print ketv
    #np.transpose(brav[(i+j)%size]).dot(ketv[(i+j)%size])
    for i in range (0,size):
        product = 1
        for j in range (0,size-2):
            product = product * np.transpose(brav[(i+j)%size]).dot(ketv[(i+j)%size])
        a = (i-2)%size
        b = (i-1)%size
        interactionx = (-0.5)*(np.transpose(brav[a]).dot(sigx).dot(ketv[a]))*(np.transpose(brav[b]).dot(sigx).dot(ketv[b]))
        interactiony = (-0.5)*(np.transpose(brav[a]).dot(sigy).dot(ketv[a]))*(np.transpose(brav[b]).dot(sigy).dot(ketv[b]))
        interactionz = (-0.5)*(np.transpose(brav[a]).dot(sigz).dot(ketv[a]))*(np.transpose(brav[b]).dot(sigz).dot(ketv[b]))
        interaction = interactionx + interactiony + interactionz
        product = product * interaction
        element = element + product
    return element

#visualize a general state in 16D by a|0000> + .... + e|1111>
def visual(raw):
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    temp = raw
    print 'The vector corresponding to this state is'
    for j in range(0,dim):
        vector = states(j+1)
        print temp[j],
        print ' |',
        for i in range(0,size):
            if vector[0][i] > 0.5:
                print 1,
            else:
                print 0,
                    
        print '> +',
    print '####'

#plot the sigma z of a state
def plotsigmaZ(raw,e,k):
    x = np.array(range(1,size+1))
    y = np.zeros(size)
    for j in range (0,dim):
        vector = states(j+1)
        for i in range(0,size):
            
            if vector[0][i] > 0.5:
                y[i] = y[i] + (np.absolute(raw[j]))**2
            else:
                y[i] = y[i] - (np.absolute(raw[j]))**2
    print 'The spin z of this state is :'
    plt.clf()
    plt.xlabel('position')
    plt.ylabel(r'<$\sigma_z$>')
    plt.ylim(-1,1)
    plt.title('Low energy state E = '+ str(e))
    plt.plot(x,y,'ro')
    name = 'C:\ProSpin\pic\N='+str(size)+'_E='+str(e)+'_#'+str(k)+'.png'
    plt.savefig(name)
    #plt.show()

#calculate the sigma z of a state
def sigmaZ(raw):
    y = np.zeros(size)
    for j in range (0,dim):
        vector = states(j+1)
        for i in range(0,size):
            
            if vector[0][i] > 0.5:
                y[i] = y[i] + (np.absolute(raw[j]))**2
            else:
                y[i] = y[i] - (np.absolute(raw[j]))**2
    return y

#calculate the energy average corresponds to T and plot <sigma z> at T
def denT(eigenvalues,T,vvv):
    E = 0
    Z = 0
    x = np.array(range(1,size+1))
    y = np.zeros(size)
    tmp = 0
    #density = np.zeros((dim,dim))
    for i in range(0,dim):
        tmp = math.exp(-eigenvalues[i]/T)
        #density[i][i] = tmp
        E = E + eigenvalues[i] * tmp
        Z = Z + tmp
        for j in range(0,size):
            y[j] = y[j] + sigmaZ(vvv[i])[j] * tmp
    for k in range(0,size):
        y[k] = y[k]/Z
    print 'The average spin z when T = ',T,' is: '
    plt.ylim(-1,1)
    plt.plot(x,y,'rx')
    plt.xlabel('position')
    plt.ylabel('<spin z>')
    plt.show()    
    Eav = E/Z
    print 'The average energy of T = ',T,' is <E>(T) = ', Eav.real

#calulate the energy average correspond to T
def EvT(eigenvalues,T):
    E = 0
    Z = 0
    for i in range(0,dim):
        tmp = math.exp(-eigenvalues[i]/T)
        E = E + eigenvalues[i] * tmp
        Z = Z + tmp
    Eav = E/Z
    return Eav

#plot Energy average T from 0 to 20
def PlotEvT(eigenvalues):
    tmp = np.zeros(100)
    step=0
    for i in range (0,100):
        tmp[i] = step
        step = step + 0.2
    E = np.zeros(100)
    for k in range (0,100):
        E[k] = EvT(eigenvalues,tmp[k])
    plt.xlabel('Temperature')
    plt.ylabel('Energy Average')
    plt.plot(tmp,E,'rx')
    plt.show()    


#Plot a histogram of energy spectrum
def spechisto(eigenvalues):
    plt.hist(eigenvalues.real, 20, normed=1, facecolor='green', alpha=0.75)
    plt.xlabel('energy')
    plt.ylabel('# of states')
    plt.grid(True)
    plt.show()
    
#######################above is the framework of a spin chain################################

#dealing with N>8 ,too slow for exact diagonalization
def lanzcos():
    print 'haha'

#observable n(k) defined as ...   
def nk():
    print 'haha'

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

print 'Now calculating the matrix elements...'
for i in range(0,dim):
    for j in range(0,dim):
        energy[i][j] = matelt(i+1,j+1)
#w,v = LA.eig(energy)



print energy
start1 = time.time()
print 'Time to compute matrix element :',start1-start0,' s'


print "#############eigenvalues########################"
#w,v = LA.eig(energy)
#vT=np.transpose(v)
#idx = w.real.argsort()
#wnew = w[idx]
#vnew = v[:,idx]
#vTnew = np.transpose(vnew)

#print 'The eigenvalues are ', wnew.real
#for i in range(0,dim):
    #print 'The',i,'Th state of eigenenergy = ',w[i],' The eigenvector is '
    #print vT[i]



print '#############################################'
#visual(vT[4])
#for i in range(0,10):
    #plotsigmaZ(vT[i])
#spechisto(w)

#for i in range (0,8):
    #print 'For the state with energy E = ',wnew[i].real,' :'
    #plotsigmaZ(vTnew[i])

#denT(wnew.real,T,vT)

print 'Now using the lanczos algorithm..............'
h,j = sla.eigs(energy,50,which='SR')
idx = h.real.argsort()
hnew = h[idx]
jnew = j[:,idx]
jTnew = np.transpose(jnew)


#print hnew.real
#for i in range (0,50):
    #print 'For the state with energy E = ',hnew[i].real,' :'
    #plotsigmaZ(jTnew[i],hnew[i].real,i)

start2 = time.time()
print 'Time to compute sparse eigenvalues :',start2-start1 ,' s'
