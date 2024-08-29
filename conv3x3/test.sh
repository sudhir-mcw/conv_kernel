rm *.bin *.npy *.o
g++ conv64x128x3x3_nchw.cc -o conv64x128x3x3_nchw.o  -std=c++11 -lcnpy && ./conv64x128x3x3_nchw.o
g++ conv64x128x3x3_nhwc.cc -o conv64x128x3x3_nhwc.o  -std=c++11 -lcnpy && ./conv64x128x3x3_nhwc.o
/home/ubuntu/tvm_env/bin/python conv64x128x3x3_nchw.py
/home/ubuntu/tvm_env/bin/python conv64x128x3x3_nhwc.py
echo "NCHW"
/home/ubuntu/tvm_env/bin/python compare.py conv64x128x3x3_nchw.npy py_conv64x128x3x3_nchw.npy
echo "NHCW"
/home/ubuntu/tvm_env/bin/python compare.py conv64x128x3x3_nhwc.npy py_conv64x128x3x3_nhwc.npy
rm *.bin *.npy *.o