INCLUDE=-I/usr/local/cuda/8.0/cuda/include 

SOURCE=dense_matvec.cu
EXECUTABLE=dense

#SOURCE=spmv.cu
#EXECUTABLE=sparse

$(EXECUTABLE): $(SOURCE)
	nvcc $(INCLUDE) $< -o $@ 

clean:
