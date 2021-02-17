**Program arguments** : ../pi-text.txt  
**Need Microsoft MPI & MPI SDK Installed** : https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi

Build with Cmake, then go in the directory where *.exe is located.  
```shell
cd ./cmake-build-debug/
```
To execute with sequentially use :
```shell
TP4.exe
```

To execute with MPI use :
```shell
mpiexec -np 8 TP4.exe
```

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