from sockets import Driver, InterfaceSocket, Status
import os
import numpy as np
import time
import sys
from aptnn.atom import Atom, Frame
from aptnn.box import Box
from datetime import datetime
import predict_functions as pf
from aptnn.io.xyz import Trajectory, write_conf
from mpi4py import MPI
import pickle
import warnings
import psutil

def get_cpu_memory_info():
    mem = psutil.virtual_memory()
    total = mem.total
    available = mem.available
    used = total - available
    return available, used, total

def read_element_symbols(filename):
    element_symbols=[]

    with open(filename, 'r') as f:
        num_atoms=int(f.readline().strip())
        #now read the comment line  
        comment_line=f.readline().strip()
        #now read the element symbols
        for i in range(num_atoms):
            element_symbols.append(f.readline().strip().split()[0])
    return element_symbols

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    driver = None
    frame = None
    #open up the socket server
    if rank == 0:
        server = InterfaceSocket()
        server.open()
        ele_symbs=read_element_symbols('pos.xyz')
    frame_count =0
    net = pf.load_committee_aptnn('c-aptnn.torch')
    print('loaded comittee',flush=True)
    box = pf.load_box('box')

    if rank ==0 and (driver is None or driver.status==Status.Disconnected):
        with open('ready.txt', 'w') as f:
            f.write('ready')
        try:
            client, address = server.server.accept() #implement a timeout here with the accept
            driver = Driver(client)
            comm.bcast(0, root=0)
        except: 
            #add a time out for 10 mins here, if timeout exceeded then broadcast status codes like 0 or 1
            #if other codes get a 1 then exit the program
            comm.bcast(1, root=0)
            exit(1)
    else:
        ret = comm.bcast(None, root=0)
        if ret == 1:
            exit(1)
    
    if rank == 0:
        while True:
            start = time.time()
            print(f'server1,{datetime.now()}',flush=True)
            try: 
                A = driver.get_data() #if this returns error, if exceeded then broadcast a none and then exit with 1
                print('got the data',flush=True)
            except Exception as e:
                print(f'Error: {e}')
                net.predict(None)
                exit(1)
            print(f'server2,{datetime.now()}',flush=True)
            #now we have element_symbols and coords, stored as ele_symbs and A
            #turn this into atoms and then frame
            #atoms = [Atom(symbol, coord) for symbol, coord in zip(elements, coords)]
            atoms=[Atom(symbol, coord) for symbol, coord in zip(ele_symbs, A)]
            frame_count += 1
            frame = Frame(atoms, frame_count, box)
            print(f'server3,{datetime.now()}',flush=True)
            if frame is not None:
                result=net.predict([frame])
                end = time.time()
                print(f'Time taken to predict: {end-start}',flush=True)
                #send the data back to the client
                data_as_arr=result['apt'][0]
                summedtensors=np.sum(data_as_arr,axis=0)
                data_as_arr=[i-summedtensors/len(data_as_arr) for i in data_as_arr]
                print(np.sum(data_as_arr,axis=0))
                var_as_arr=result['std'][0]
                tottensordata=np.concatenate((data_as_arr,var_as_arr),axis=0)
                print(result['apt'][0][-1],flush=True)
                print(result['std'][0][-1],flush=True)
                driver.send_data(tottensordata)
                frame = None
                print(f'server4,{datetime.now()}',flush=True)
            else:
                print('frame is none, skipping predict section',flush=True)
    else:
        while net.predict([]):
            pass

if __name__ == '__main__':
    warnings.filterwarnings("ignore")   
    main()


