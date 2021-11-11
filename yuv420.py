import numpy as np

def readYUV420(name: str, resolution: tuple, upsampleUV: bool = False):
    height = resolution[0]
    width = resolution[1]
    bytesY = int(height * width)
    bytesUV = int(bytesY/4)
    Y = []
    U = []
    V = []
    with open(name,"rb") as yuvFile:
        while (chunkBytes := yuvFile.read(bytesY + 2*bytesUV)):
            Y.append(np.reshape(np.frombuffer(chunkBytes, dtype=np.uint8, count=bytesY, offset = 0), (width, height)))
            U.append(np.reshape(np.frombuffer(chunkBytes, dtype=np.uint8, count=bytesUV, offset = bytesY),  (width//2, height//2)))
            V.append(np.reshape(np.frombuffer(chunkBytes, dtype=np.uint8, count=bytesUV, offset = bytesY + bytesUV), (width//2, height//2)))
    Y = np.stack(Y)
    U = np.stack(U)
    V = np.stack(V)
    if upsampleUV:
        U = U.repeat(2, axis=1).repeat(2, axis=2)
        V = V.repeat(2, axis=1).repeat(2, axis=2)
    return Y, U, V


def readYUV420Range(name: str, resolution: tuple, range: tuple, upsampleUV: bool = False):
    height = resolution[0]
    width = resolution[1]
    bytesY = int(height * width)
    bytesUV = int(bytesY/4)
    Y = []
    U = []
    V = []
    with open(name,"rb") as yuvFile:
        if range:
            startLocation = range[0]
            endLocation = range[1] + 1
            startLocationBytes = startLocation * (bytesY + 2*bytesUV)
            endLocationBytes = endLocation * (bytesY + 2*bytesUV)
            data = np.fromfile(yuvFile, np.uint8, endLocationBytes-startLocationBytes, offset=startLocationBytes).reshape(-1,bytesY + 2*bytesUV)
            Y = np.reshape(data[:, :bytesY], (-1, width, height))
            U = np.reshape(data[:, bytesY:bytesY+bytesUV], (-1, width//2, height//2))
            V = np.reshape(data[:, bytesY+bytesUV:bytesY+2*bytesUV], (-1, width//2, height//2))
        else:
            while (chunkBytes := yuvFile.read(bytesY + 2*bytesUV)):
                Y.append(np.reshape(np.frombuffer(chunkBytes, dtype=np.uint8, count=bytesY, offset = 0), (width, height)))
                U.append(np.reshape(np.frombuffer(chunkBytes, dtype=np.uint8, count=bytesUV, offset = bytesY),  (width//2, height//2)))
                V.append(np.reshape(np.frombuffer(chunkBytes, dtype=np.uint8, count=bytesUV, offset = bytesY + bytesUV), (width//2, height//2)))
            Y = np.stack(Y)
            U = np.stack(U)
            V = np.stack(V)
    if upsampleUV:
        U = U.repeat(2, axis=1).repeat(2, axis=2)
        V = V.repeat(2, axis=1).repeat(2, axis=2)
    return Y, U, V
