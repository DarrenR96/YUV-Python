import numpy as np

def readYUVtoArr(file,width = 1280,height = 720):
    px = width*height
    YUV = np.fromfile(file,dtype='uint8')
    frames = int(YUV.shape[0]/(px+((2*px)/4)))
    Y = []
    U = []
    V = []
    for frame in range(0,frames):
        Y.append(YUV[int((px + 2*(px/4))*frame):int((px + 2*(px/4))*frame+px)].reshape(height,width))
        U.append(YUV[int((px + 2*(px/4))*frame+px):int((px + 2*(px/4))*frame+px+(px/4))].reshape(height//2,width//2))
        V.append(YUV[int((px + 2*(px/4))*frame+px+(px/4)):int((px + 2*(px/4))*frame+px+2*(px/4))].reshape(height//2,width//2))
    return np.array(Y), np.array(U), np.array(V)
