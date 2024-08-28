rm *.bin *.npy conv64x128x1x1
g++ conv64x128x1x1.cc -o conv64x128x1x1  -std=c++11 -lcnpy && ./conv64x128x1x1
g++ conv64x128x1x1_nhwc.cc -o conv64x128x1x1_nhwc  -std=c++11 -lcnpy && ./conv64x128x1x1_nhwc
/home/ubuntu/tvm_env/bin/python conv64x128x1x1.py
echo "NCHW"
/home/ubuntu/tvm_env/bin/python compare.py conv64x128x1x1_nchw.npy py_conv64x128x1x1.npy
echo "NHCW"
/home/ubuntu/tvm_env/bin/python compare.py conv64x128x1x1_nhwc.npy py_conv64x128x1x1.npy
