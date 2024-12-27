CC="C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\ifx.exe"

all: main.f90 dataset.f90 tensor.f90
	$(CC) -O3 -flto -fuse-ld=lld -4Yportlib $^ -o main.exe

clean:
	del main.exe