# YUV-Python
Simple function for reading in YUV 420 Files in Python

## Usage
Takes in path to YUV file and resolution of the file (eg. 1920x1080) and returns three numpy arrays, one for Y, U and V channels of dimensions (# of frames * height * width) for Y and (# of frames * height/2 * width/2) for U and V. You can also upsample to include full sized U and V arrays.


```python
from yuv420 import readYUV420

Yarr, Uarr, Varr = readYUVtoArr("path/file.yuv", (1920,1080), upsampleUV=True)
```
