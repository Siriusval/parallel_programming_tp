# PPAR TP4 - Game of Life

## Preparation

### Install MPI

- On Windows :  
  **Need Microsoft MPI & MPI SDK Installed** : https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi
- On Linux :

```shell
sudo apt-get install mpich
```

### Create makefiles with Cmake (in parent directory)

```shell
cd TP4
mkdir cmake-build-debug
cmake -S ./ -B ./cmake-build-debug
```

### Build executables

```shell
make
```

It will build 2 differents executables :

- TP4_Seq for sequential execution.
- TP4_MPI for parallel execution w/ MPI.

### Run

Go in the directory where \*.exe is located

```shell
./TP4_Seq.exe
```

To execute with MPI use :

```shell
mpiexec -np 8 ./TP4_MPI.exe
```

#### Choice for mpiexe -np argument

Matrix width & height is 32.  
You can use this as -np argument

| nbProc | nbRow per proc |
| ------ | -------------- |
| 1      | 32             |
| 2      | 16             |
| 4      | 8              |
| 8      | 4              |
| 16     | 2              |
| 32     | 1              |
