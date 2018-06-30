ARMNN_LIB = ../armnn
ARMNN_INC = ../armnn/include

armnn_caffe: armnn_caffe.cpp armnn_loader.hpp
	arm-linux-gnueabihf-g++ -O3 -std=c++14 -I$(ARMNN_INC) armnn_caffe.cpp -o armnn_caffe -L$(ARMNN_LIB) -larmnn -larmnnCaffeParser

clean:
	-rm -f armnn_caffe

test: armnn_caffe
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(ARMNN_LIB) ./armnn_caffe