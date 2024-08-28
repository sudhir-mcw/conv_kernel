rm *.bin *.npy conv64x128x3x3 conv64x128x3x3_nhwc
g++ conv64x128x3x3.cc -o conv64x128x3x3  -std=c++11 -lcnpy && ./conv64x128x3x3
g++ conv64x128x3x3_nhwc.cc -o conv64x128x3x3_nhwc  -std=c++11 -lcnpy && ./conv64x128x3x3_nhwc
/home/ubuntu/tvm_env/bin/python conv64x128x3x3.py
echo "NCHW"
/home/ubuntu/tvm_env/bin/python compare.py conv64x128x3x3_nchw.npy py_conv64x128x3x3.npy
echo "NHWC"
/home/ubuntu/tvm_env/bin/python compare.py conv64x128x3x3_nhwc.npy py_conv64x128x3x3.npy