import struct
import numpy as np
import math as m

##
#def cart2sph(x,y,z):
#    XsqPlusYsq = x**2 + y**2
#    r = m.sqrt(XsqPlusYsq + z**2)               # r
#    elev = m.atan2(z,m.sqrt(XsqPlusYsq))     # theta
#    az = m.atan2(y,x)                           # phi
#    return r, elev, az

def mainfn():
	tff = raw_input("Transfer function file to be used: ")
	with open('/home/akshay/anaconda/ModulesPython/MUSICSEVD/{}'.format(tff),'rb') as data:
		HARK = struct.unpack('<4B',data.read(4))
		print HARK," is 'HARK'"
		SIZE = struct.unpack('<I',data.read(4))
		print SIZE," is the SIZE"
		HGTF = struct.unpack('<4B',data.read(4))
		print HGTF," is 'HGTF'"
		
	
		# fmt subchunk
	
		fmt = struct.unpack('<4B',data.read(4))
		print fmt, " is 'fmt<space>'"	
		sizeofsubchunk = struct.unpack('<I',data.read(4))	
		print sizeofsubchunk, " is the size of subchunk"	
		fmtversion = struct.unpack('<2H',data.read(4))
		print fmtversion, " is the fmt version"	
		formatid = struct.unpack('<I',data.read(4))
		print formatid, " is the format ID: (0: undefined, 1: M2PG, 2: GTF, 3: SM)"	
		nmics  = struct.unpack('<I',data.read(4))
		print nmics, " is the number of microphones"	
		fftlength = struct.unpack('<I',data.read(4))
		print fftlength, " is the fft length"	
		implen = struct.unpack('<I',data.read(4))
		print implen, " is the impulse response length"	
		fs = struct.unpack('<I',data.read(4))
		print fs, " is the sampling frequency[Hz]"
		print "Coordinates of microphone position[m]"
		x1,y1,z1 = struct.unpack('<3f',data.read(12))
		print "x1 = ",x1," y1 = ",y1," z1 = ",z1
		x2,y2,z2 = struct.unpack('<3f',data.read(12))
		print "x2 = ",x2," y2 = ",y2," z2 = ",z2
		x3,y3,z3 = struct.unpack('<3f',data.read(12))
		print "x3 = ",x3," y3 = ",y3," z3 = ",z3
		x4,y4,z4 = struct.unpack('<3f',data.read(12))
		print "x4 = ",x4," y4 = ",y4," z4 = ",z4
	
	
		# M2PG SUBCHUNK
	
		m2pg = struct.unpack('<4B',data.read(4))
		print m2pg," is 'M2PG'"
		M2PGsize = struct.unpack('<I',data.read(4))
		print M2PGsize," is the M2PG subchunk SIZE"
		m2pgversion = struct.unpack('<2H',data.read(4))
		print m2pgversion, " is the M2PG version(major, minor)"
		Nd = struct.unpack('<I',data.read(4))
		print Nd, "Number of elements in horizontal direction (Nd)"
		Nr = struct.unpack('<I',data.read(4))
		print Nr, "Number of elements of distance (Nr)"
		# Getting the coordinates of the source positions of the 72 source directions
		print "The source locations azimuth angles are : \n"
		srcloc = np.zeros((Nd[0],1,3),dtype = float)
		az = np.zeros((Nd[0],1,1),dtype = float)
		for d in range(Nd[0]):
			srcloc[d,0,:] = struct.unpack('<3f',data.read(12))
			az[d,0,0] = m.atan2(srcloc[d,0,1],srcloc[d,0,0]) * 180/m.pi
			print az[d,0,0]
		
		
		## Transfer function <complex<float>>
		tf = np.zeros((Nd[0],Nr[0],nmics[0],(fftlength[0]/2+1)),dtype = complex)
		#tf = []	
		row_fmt = '<f'
		print "row_fmt is ", row_fmt
		row_len = struct.calcsize(row_fmt)
		print "row_len is : ", row_len
		print "Nd[0]  is " ,Nd[0]
		for theta in range(Nd[0]):
                        for i in range(nmics[0]):
                                for fy in range(fftlength[0]/2 +1):
                                        #tf[theta,0,i,fy] = struct.unpack('<f',data.read(4))
                                        reals = struct.unpack(row_fmt, data.read(row_len))
                                        imags = struct.unpack(row_fmt, data.read(row_len))
                                        tf[theta,0,i,fy] = complex(reals[0],imags[0])
                                #print theta		
				#print str(tf.shape)
				#print tf
		SPEC = struct.unpack('<4B',data.read(4))	
		print SPEC, " is SPEC."	
		print Nd[0]	
		return tf,az,Nd[0]
	
if __name__ == '__main__':
	x = mainfn()
