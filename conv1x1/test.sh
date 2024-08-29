rm *.bin *.npy *.o
g++ conv64x128x1x1.cc -o conv64x128x1x1.o  -std=c++11 -lcnpy && ./conv64x128x1x1.o
g++ conv64x128x1x1_nhwc.cc -o conv64x128x1x1_nhwc.o  -std=c++11 -lcnpy && ./conv64x128x1x1_nhwc.o
/home/ubuntu/tvm_env/bin/python conv64x128x1x1.py
/home/ubuntu/tvm_env/bin/python conv64x128x1x1_nhwc.py
echo "NCHW"
/home/ubuntu/tvm_env/bin/python compare.py conv64x128x1x1_nchw.npy py_conv64x128x1x1.npy
echo "NHCW"
/home/ubuntu/tvm_env/bin/python compare.py conv64x128x1x1_nhwc.npy py_conv64x128x1x1_nhwc.npy
rm *.bin *.npy *.o