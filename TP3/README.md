---
title: Parallel Programming TP3 - MPI  
author: vhulot  
date: 2021-01
---

# Parallel Programming TP3 : MPI

MPI Introduction.
- Count characters in text and parallelize work between multiple workers.


[**Course description**](https://istic.univ-rennes1.fr/ue-ppar)

Exercises for University.

## Setup
**Need Microsoft MPI & MPI SDK Installed** :  
https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi

## Program arguments
```shell
../pi-text.txt
``` 

## Execute
To execute with MPI, build then use :
```shell
mpiexec -np 12 TP3.exe ../pi-text.txt
```