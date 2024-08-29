rm *.bin *.npy *.o
g++ conv64x128x1x1_nchw.cc -o conv64x128x1x1_nchw.o  -std=c++11 -lcnpy && ./conv64x128x1x1_nchw.o
g++ conv64x128x1x1_nhwc.cc -o conv64x128x1x1_nhwc.o  -std=c++11 -lcnpy && ./conv64x128x1x1_nhwc.o
/home/ubuntu/tvm_env/bin/python conv64x128x1x1_nchw.py
/home/ubuntu/tvm_env/bin/python conv64x128x1x1_nhwc.py
echo "NCHW"
/home/ubuntu/tvm_env/bin/python compare.py conv64x128x1x1_nchw.npy py_conv64x128x1x1_nchw.npy
echo "NHCW"
/home/ubuntu/tvm_env/bin/python compare.py conv64x128x1x1_nhwc.npy py_conv64x128x1x1_nhwc.npy
rm *.bin *.npy *.o