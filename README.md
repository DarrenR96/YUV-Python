# YUV-Python
Simple function for reading in YUV 420 Files in Python

## Usage
Takes in path to YUV file and resolution of the file (eg. 1280x720) and returns three numpy arrays, one for Y, U and V channels of dimensions (# of frames * height * width) for Y and (# of frames * height/2 * width/2) for U and V.


```python
from yuv420 import readYUVtoArr

Yarr, Uarr, Varr = readYUVtoArr("path/file.yuv", width=1280, height=720)
```
