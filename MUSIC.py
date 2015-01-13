#Localization using MUSIC-SEVD algorithm
#Localization is done OFFLINE

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import math
import readingdat
import powerspec

fs = 16000
nc = 4
M = 160
N = 512

def blockd(v,N,M):  #Separating the vector into blocks
    n = len(v)
    maxblockstart = n - N + 1
    lastblockstart = maxblockstart - ((maxblockstart-1) % M)
    numblocks = (lastblockstart-1)/M + 1

    f = np.zeros((numblocks,N))

    for i in range(numblocks):
        for j in range(N):
            f[i,j] = v[(i-1)*M+j]
  
    return f

def nblocks(v,N,M):
    n = len(v)
    maxblockstart = n - N + 1
    lastblockstart = maxblockstart - ((maxblockstart-1) % M)
    return (lastblockstart-1)/M + 1
    
def mfft(frame):    #frame has nc number of rows
    #print "Shape of each frame:" + str(frame.shape)
    #print "Now calculating the fft for each frame."
    N = frame.shape[1]
    nc = frame.shape[0]
    fftmat = np.zeros((nc,N),dtype=complex)
    fftmat2 = np.zeros((nc,N/2 + 1),dtype = complex)
    #plt.figure(1)

    ##freq = np.fft.fftshift(np.fft.fftfreq(N))
    #print frame[i,:].shape
    #print freq.shape
    ##freq2 = freq[:(N/2+1)]
    #freq2hz = abs(freq2*fs)  #Freq in Hz
    #freq2hz = freq2hz[::-1]
    #print freq2.shape
    
    #print freq2hz
    for i in range(nc):
        fftmat[i,:] = np.fft.fftshift(np.fft.fft(frame[i,:],N))
        
        fftmat2[i,:] = fftmat[i,:(N/2+1)]
##        print "FFT of channel: ",str(i) 
##        print fftmat2.shape
##        
##        plt.subplot(nc,1,i)
##        plt.plot(freq2hz,fftmat2[i,:].real,freq2hz,fftmat2[i,:].imag)
##       
##    plt.show()
    fftmat2 = np.conjugate(fftmat2[:,::-1])
    #print fftmat2.shape
    return fftmat2  #Returning fft with zero freq as the first element

def Rmatrix(fftmat):  # fftmat is the multi-fft of ONE frame
    #print "Shape of Multi FFT: " + str(fftmat.shape)    #Should be 4x257
    R = np.zeros((nc,nc,(N/2 +1)),dtype=complex)
    #print "Shape of R matrix: " + str(R.shape)  #Should be 4x4x257
    for fy in range(N/2 + 1):
##        print fy
##        print fftmat[:,fy]      #fftmat[:,fy] is an array.
##        print "shape of fftmat[:,fy]"   #When converted to matrix --> [1x4] matrix
##        print fftmat[:,fy].shape
        x = np.conjugate(fftmat[:,fy])
##        print "printing x"
##        print x.shape
        y=np.matrix(np.transpose(x))
##        print "printing y"
##        print y.shape
##
##        print np.matrix(fftmat[:,fy]).shape
        p = np.transpose(np.matrix(fftmat[:,fy]))   #Matrix fftmat[:,fy] has to be [4x1]
##      print p.shape
        R[:,:,fy] = p*y
##        print "Shape of R: " + str(R.shape)
##        print "R matrix: ", R[:,:,fy]
    # R is the Correlation matrix for each frequency, and for THIS frame.
    return R    
    
    


def mainfn():

    #Read the file
    wavfil = raw_input("Wav file to be read: ")
    f = wavfile.read('/home/akshay/anaconda/ModulesPython/MUSICSEVD/{}'.format(wavfil))
    rdata = f[1]
    nc = rdata.shape[1]
    print "Number of channels: " + str(nc)
    fs = f[0]
    print "Sampling frequency: " + str(fs)
    (Pxx, freqs, bins, im) = plt.specgram(rdata[:,0], NFFT = 512, Fs = fs,noverlap=160)
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency in Hz')
    plt.show()

    tf,az,Nd = readingdat.mainfn()
    #print tf
    #print str(tf.shape)

    # INPUTS:
    N = 512     # Block Size & N point FFT
    M = 160     # Block Increment
    WINDOW = 50     # Time averaging of the CM, WINDOW_TYPE is FUTURE


    ## Plot the audio signal - Channel 1
    ##xaxis = np.arange(len(rdata))
    ##plt.plot(xaxis,rdata[:,0])
    ##plt.title('Channel 1: First microphone')
    #plt.show()


    # Divide into frames with overlap - Use blockd
    ## Assumption: rdata has channels on columns, and samples on rows

    rdata = np.transpose(rdata) #Comment out if samples on col. and channels on rows
    num_blocks = nblocks(rdata[0,:],N,M)

    blockeddata = np.zeros((nc,num_blocks,N))# blockeddata is a 3D matrix

    for i in range(nc):
        blockeddata[i,:,:] = blockd(rdata[i,:],N,M)
    print "Shape of blocked data: " + str(blockeddata.shape)

    #EACH FRAME COMPUTATION:
    #Each frame has to be taken for FFT

##    fftmat = mfft(blockeddata[:,0,:])
     
    fnum = 0    #fnum is the frame index
    #The correlation matrix is calculated once every PERIOD number of frames.
    # Time averaging is done with WINDOW_TYPE = FUTURE
    #print "Calculating the time-averaged correlation matrices"

    
    for t in range(int(math.ceil(num_blocks/WINDOW))):
        #fnum is Frame index
        i=0
        R_tot = np.zeros((nc,nc,N/2 + 1),dtype = complex)
	print "Localization for frames: ", str(t*50)," to ",str((t+1)*50 - 1)
        while i < WINDOW:
            if fnum >= num_blocks:
                break
            else:
                fftmat = mfft(blockeddata[:,fnum,:])
                #print fftmat
                #print "Shape of fftmat: " + str(fftmat.shape)
                R_tot = R_tot + Rmatrix(fftmat)
                fnum = fnum+1
                i = i+1
##        print i
##        print fnum
        R_avg = R_tot/(i)
        print "R_avg matrix for freq bin 20: "
        print R_avg[:,:,20]
##        print "Shape of R_avg is: " + str(R_avg.shape)
        
	#Eigen value decomposition is done for each frequency 
    	w = np.zeros((nc,N/2 + 1),dtype = complex)
    	v = np.zeros((nc,nc,N/2 + 1),dtype = complex)
    	for fy in range(N/2 + 1):
        	w[:,fy],v[:,:,fy] = np.linalg.eig(R_avg[:,:,fy])
        	
        	#print "Shape of w: " + str(w.shape)
        	#print "Shape of v: " + str(v.shape)

    	print "values: " , w[:,5]		#printing the 3rd freq bin's eigen values and eigen vectors
    	#print w[:,2].shape
    	print "vectors: ", v[:,:,5]
##        print "v shape: ", str(v[:,1,2].shape)	

##    Calculation of MUSIC spectrum
        print "Calculating MUSIC spectrum"
        powerspec.mainfn(tf,v,t,az,Nd)
        	

        
if __name__ == '__main__':
    mainfn()
