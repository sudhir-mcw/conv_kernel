rm *.bin *.npy conv64x128x3x3
g++ conv64x128x3x3.cc -o conv64x128x3x3  -std=c++11 -lcnpy && ./conv64x128x3x3
/home/ubuntu/tvm_env/bin/python conv64x128x3x3.py
/home/ubuntu/tvm_env/bin/python compare.py conv64x128x3x3.npy py_conv64x128x3x3.npy