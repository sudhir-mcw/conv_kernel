rm *.bin *.npy conv64x128x1x1
g++ conv64x128x1x1.cc -o conv64x128x1x1  -std=c++11 -lcnpy && ./conv64x128x1x1
/home/ubuntu/tvm_env/bin/python conv64x128x1x1.py
/home/ubuntu/tvm_env/bin/python compare.py conv64x128x1x1.npy py_conv64x128x1x1.npy