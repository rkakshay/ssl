import numpy as np
import math
import matplotlib.pyplot as plt

fs = 16000
nc = 4
M = 160
N = 512
Nr = 1
Ns = 1  #Ns - number of sources
fmin = 800
fmax = 5000

def mainfn(tf,v,count,az,Nd):

## Numerator -- Power spectrum calculation
##    print "entered! "
##    print str(tf.shape)     #[72 x 1 x 4 x 257]  
##    print str(tf.shape[3])  #[257]
##    print str(v.shape)      #[4 x 4 x 257]
    Nps = np.zeros((Nd,(N/2 +1),1),dtype=complex)   #Numerator
    Dps = np.zeros((Nd,(N/2 +1),1),dtype=complex)   #Denominator
    Powspec = np.zeros((Nd,(N/2 +1),1),dtype=complex)   #Result
    
    # tf -- [72 x 1 x 4 x 257]
    for theta in range(tf.shape[0]):
        for fy in range(tf.shape[3]):
            
##            print str(tf[theta,0,:,fy].shape)
##            print tf[theta,0,:,fy]
            x = np.conjugate(tf[theta,0,:,fy])
##            print "printing x"
##            print x
            y=np.matrix(np.transpose(x))
##            print "printing y"
##            print y
##            print y.shape
##
##            print np.matrix(fftmat[:,fy]).shape
            p = np.transpose(np.matrix(tf[theta,0,:,fy]))   #Matrix tf[theta,0,:,fy] has to be [4x1]
##            print p.shape
##            print (y*p).shape
            j = np.array(y*p)
##            print "printing j shape", j[0].shape
##            ans = np.absolute(j[0])
##            print "printing ans", ans
            Nps[theta,fy,0] = np.absolute(j[0])
##            print "Shape of Nps: " + str(Nps.shape)


## Denominator -- Power spectrum calculation

    
    for theta in range(tf.shape[0]):
        for fy in range(tf.shape[3]):
            x = np.conjugate(tf[theta,0,:,fy])
            y = np.matrix(np.transpose(x))
            for i in range(Ns, nc):
                #print str(v[:,i,fy].shape)
                vmat = np.transpose(np.matrix(v[:,i,fy]))
                p = np.array(y*vmat)
                #print str(p.shape)  #Should be [1x1]
                #print np.absolute(p[0])
                Dps[theta,fy,0] = Dps[theta,fy,0] + float(np.absolute(p[0]))
                #print str(Dps.shape)

## Result -- Power spectrum calculation
    #print "Denominators done."
    for theta in range(tf.shape[0]):
        for fy in range(tf.shape[3]):
            Powspec[theta,fy,0] = Nps[theta,fy,0]/Dps[theta,fy,0]
        

## Broadband integration
            
    
    #print "Shape of Power spectrum", str(Powspec.shape)     # Should be [72 x 257 x 1]
    Ps = np.zeros((Nd,1),dtype = float)      #Contains the sum of the power spectrum for each angle
    r = np.zeros((N/2 +1,1),dtype = float)     #Contains the power spectrum for each frequency
    #print str(Powspec[1].shape)
    for ang in range(Nd):
        r = Powspec[ang,:,:]
        Ps[ang,:] = np.sum(r,axis=0)
    #print "shape of Ps: ", str(Ps.shape)    #[72 x 1]  
    #print "shape of r: ", str(r.shape)     #[257 x 1] 
    #xaxis = np.arange(Nd)*(360/Nd) -180	#For scaling to the coordinate system
    xaxis = az[:,0,0]
    #print "xaxis: ", xaxis
    #xxaxis = np.fft.fftshift(xaxis)
    #print "xxaxis: ", xxaxis
    #print str(xaxis.shape)
    if xaxis[:1] < 0:
    	plt.plot(xaxis,Ps)
    	plt.xlabel('Degrees')
    	plt.ylabel('MUSIC Power spectrum')
    	plt.title('Localization result for frames: {} to {}'.format(str(count*50),str((count+1)*50 - 1)))
	psorted = sorted(enumerate(Ps[:,0]), key=lambda x:x[1], reverse = True)
        print "Angle -- MUSIC Power spectrum"
    	for item in psorted[:4]:
        	print str(item[0]*(360/Nd)-180)," -- ", str(item[1])
    	plt.show()
    	return()
    else:
	xaxis = np.arange(Nd)*360/Nd
        plt.plot(xaxis,Ps)
    	plt.xlabel('Degrees')
    	plt.ylabel('MUSIC Power spectrum')
    	plt.title('Localization result for frames: {} to {}'.format(str(count*50),str((count+1)*50 - 1)))
	psorted = sorted(enumerate(Ps[:,0]), key=lambda x:x[1], reverse = True)
        print "Angle -- MUSIC Power spectrum"
    	for item in psorted[:4]:
        	print str(item[0]*(360/Nd))," -- ", str(item[1])
    	plt.show()
    	return()
if __name__ == '__main__':
    mainfn()
