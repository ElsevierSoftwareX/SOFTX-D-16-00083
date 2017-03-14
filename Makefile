OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
    CXX = clang++
else
    CXX = g++
endif
CXXFLAGS = -Wall -O3 -std=c++11
EXECNAME=XaNSoNS
LDFLAGS = 
ifeq ($(OpenMP),1)
    CXX = g++
    CXXFLAGS += -fopenmp -D UseOMP
    LDFLAGS += -fopenmp
ifneq ($(DARWIN),)
    CXXFLAGS += -static-libgcc -static-libstdc++
    LDFLAGS += -static-libgcc -static-libstdc++
endif
    EXECNAME=XaNSoNS_OMP
endif
ifeq ($(MPI),1)
    CXX = mpicxx
    CXXFLAGS += -D UseMPI
    ifeq ($(OpenMP),1)
        EXECNAME=XaNSoNS_MPI_OMP
    else
        EXECNAME=XaNSoNS_MPI
    endif
else
ifeq ($(CUDA),1)
    NVCC = nvcc
    CXXFLAGS += -D UseCUDA
    NVCCFLAGS = -D_FORCE_INLINES -D UseCUDA -O3 -use_fast_math
    ifeq ($(TESLA),1)
        GENCODE_FLAGS = -gencode arch=compute_13,code=sm_13 -gencode arch=compute_13,code=compute_13
    else
        GENCODE_FLAGS = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52
    endif
    LDFLAGS   += -L/usr/local/cuda/lib -lcuda -lcudart_static
    EXECNAME=XaNSoNS_CUDA
else 
ifeq ($(OpenCL),1)
    CXXFLAGS += -D UseOCL
    ifneq ($(DARWIN),) 
        LDFLAGS += -framework opencl
    else
        LDFLAGS += -lOpenCL
    endif
    EXECNAME=XaNSoNS_OCL
endif
endif
endif
ODIR=$(EXECNAME)_build

all: $(EXECNAME)

ifeq ($(CUDA),1)
$(EXECNAME): $(ODIR)/main.o $(ODIR)/tinyxml2.o $(ODIR)/ReadXML.o $(ODIR)/IO.o $(ODIR)/CalcFunctions.o $(ODIR)/CalcFunctionsCUDA.o
	$(NVCC) -o $(EXECNAME) $(ODIR)/main.o $(ODIR)/tinyxml2.o $(ODIR)/ReadXML.o $(ODIR)/IO.o $(ODIR)/CalcFunctions.o $(ODIR)/CalcFunctionsCUDA.o $(LDFLAGS)
else
ifeq ($(OpenCL),1)
$(EXECNAME): $(ODIR)/main.o $(ODIR)/tinyxml2.o $(ODIR)/ReadXML.o $(ODIR)/IO.o $(ODIR)/CalcFunctions.o $(ODIR)/CalcFunctionsOCL.o
	$(CXX) -o $(EXECNAME) $(ODIR)/main.o $(ODIR)/tinyxml2.o $(ODIR)/ReadXML.o $(ODIR)/IO.o $(ODIR)/CalcFunctions.o $(ODIR)/CalcFunctionsOCL.o $(LDFLAGS)
else
$(EXECNAME): $(ODIR)/main.o $(ODIR)/tinyxml2.o $(ODIR)/ReadXML.o $(ODIR)/IO.o $(ODIR)/CalcFunctions.o
	$(CXX) -o $(EXECNAME) $(ODIR)/main.o $(ODIR)/tinyxml2.o $(ODIR)/ReadXML.o $(ODIR)/IO.o $(ODIR)/CalcFunctions.o $(LDFLAGS)
endif
endif

$(ODIR)/main.o: main.cpp typedefs.h | $(ODIR)
	$(CXX) $(CXXFLAGS) -c main.cpp -o $(ODIR)/main.o

$(ODIR)/tinyxml2.o: tinyxml2.cpp tinyxml2.h | $(ODIR)
	$(CXX) $(CXXFLAGS) -c tinyxml2.cpp -o $(ODIR)/tinyxml2.o

$(ODIR)/ReadXML.o: ReadXML.cpp typedefs.h tinyxml2.h | $(ODIR)
	$(CXX) $(CXXFLAGS) -c ReadXML.cpp -o $(ODIR)/ReadXML.o

$(ODIR)/IO.o: IO.cpp typedefs.h | $(ODIR)
	$(CXX) $(CXXFLAGS) -c IO.cpp -o $(ODIR)/IO.o

$(ODIR)/CalcFunctions.o: CalcFunctions.cpp typedefs.h | $(ODIR)
	$(CXX) $(CXXFLAGS) -c CalcFunctions.cpp -o $(ODIR)/CalcFunctions.o

$(ODIR)/CalcFunctionsCUDA.o: CalcFunctionsCUDA.cu typedefs.h | $(ODIR)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -c CalcFunctionsCUDA.cu -o $(ODIR)/CalcFunctionsCUDA.o

$(ODIR)/CalcFunctionsOCL.o: CalcFunctionsOCL.cpp typedefs.h | $(ODIR)
	$(CXX) $(CXXFLAGS) -c CalcFunctionsOCL.cpp -o $(ODIR)/CalcFunctionsOCL.o

$(ODIR):
	mkdir -p $(ODIR)

clean:
	find . -type f -name '*.o' -exec rm {} +
