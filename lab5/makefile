INCLUDE=-I/usr/local/cuda/8.0/cuda/include 

#SOURCE=dense_matvec.cu
#EXECUTABLE=dense

SOURCE=sparse_matvec.cu
EXECUTABLE=sparse

$(EXECUTABLE): $(SOURCE)
	nvcc $(INCLUDE) $< -o $@ 

clean:
