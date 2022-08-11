---
title: Parallel Programming TP4 - Game of Life MPI  
author: vhulot  
date: 2021-01
---

# Parallel Programming TP4 : Game of Life MPI

- Implementation of GoL
- Sequential & parallel versions
- linux & windows version.

[**Course description**](https://istic.univ-rennes1.fr/ue-ppar)

Exercises for University.


## Preparation

### Install MPI

- On Windows :  
  **Need Microsoft MPI & MPI SDK Installed** : https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi
- On Linux :

```shell
sudo apt-get install mpich
```

## Execute Windows/Linux (Easy Mode)

Prebuilt files are available in ./build.  
You can go directly to **Run** step.

## Execute on Windows (Hard Mode)

### Build

You can build in CLion or VisualStudio.

It will build 2 differents executables :

- TP4_Seq for sequential execution.
- TP4_MPI for parallel execution w/ MPI.

### Run

Go in the directory where \*.exe is located

```shell
cd ./cmake-build-debug/src/seq
./TP4_Seq.exe
```

To execute with MPI use :

```shell
cd ./cmake-build-debug/src/mpi
mpiexec -np 8 ./TP4_MPI.exe
```

## Execute on Linux (Hard Mode)

### Create makefiles with Cmake (in parent directory)

```shell
cd TP4
mkdir cmake-build-debug-linux
cmake -S ./ -B ./cmake-build-debug-linux
```

### Build

```shell
cd ./cmake-build-debug-linux
make
```

It will build 2 differents executables :

- TP4_Seq for sequential execution.
- TP4_MPI for parallel execution w/ MPI.

### Run

Go in the directory where \*.exe is located

```shell
cd ./src/seq
./TP4_Seq
```

To execute with MPI use :

```shell
cd ./src/mpi
mpiexec -np 8 ./TP4_MPI
```

## Choice for mpiexe -np argument

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

```

```
