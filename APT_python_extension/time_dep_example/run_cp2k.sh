#!/bin/bash

OMP_NUM_THREADS=8 mpirun -np 8 python3 server.py --host '/tmp/qiskit_apt.server.socket' > server.out 2> server.err &
sleep 15
mpirun -x OMP_NUM_THREADS=2 --map-by ppr:8:socket:PE=2 --bind-to core cp2k.psmp start.inp > cp2k.out 2> cp2k.err
