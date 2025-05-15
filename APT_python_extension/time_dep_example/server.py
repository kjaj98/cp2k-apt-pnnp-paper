import time

program_start_time = time.perf_counter()

import os
import sys
import socket
import argparse
import numpy as np

from enum import IntEnum, auto
from mpi4py import MPI
from datetime import datetime

from aptnn.atom import Atom, Frame
from aptnn.box import Box
from aptnn.committee import CommitteeAPTNN
from aptnn.io.xyz import Trajectory, write_conf


# possible messages
class EMessages(IntEnum):
    # instructs to perform prediction on the given set of coordinates
    PREDICT = 0
    # instructs to set a variable or configure a property
    SET = 1 

# possible options for variable changes
class EVariables(IntEnum):
    # instructs to open a aptnn.torch file
    APTNN_FILE = 0
    # instructs to change the employed box definition
    CELL = 1
    # instructs to set time step
    TIMESTEP = 2
    # instructs to write all atomic polar tensors to file
    WRITE_APT = 3
    # instructs to write all APT variances to file
    WRITE_VAR = 4
    # instructs to write the APT uncertainty per frame to file
    WRITE_FRAME_UNCERTAINTY = 5

# possible return codes
class EReturnCodes(IntEnum):
    # OK
    OK = 0


# GLOBAL variables
box = Box()
fout_apt = None
fout_var = None
# TODO UNTIL LINK to CP2K IS MISSING
fout_frame_uncertainty = open('frame-uncertainty.dat', 'w')

mpi = MPI.COMM_WORLD
rank = mpi.Get_rank()
timestep = 1

fortran_dt = np.dtype(float)
fortran_dt = fortran_dt.newbyteorder('<')

#############################
# Helper functions

class SocketClosedException(Exception):
    pass

def recv_buffer(conn, length):
    buf = bytearray()
    while len(buf) < length:
        tmp = conn.recv(length - len(buf))
        if not tmp:
            raise SocketClosedException()
        buf.extend(tmp)
    return buf


def recv_int(conn):
    data = recv_buffer(conn, 4)
    return int.from_bytes(data, byteorder='little')

def recv_int_mpi(conn, poll=False):
    i = None


    if rank == 0:
        try:
            i = recv_int(conn)
        except SocketClosedException:
            i = sys.maxsize

    # if True, this creates a true blocking call, where the threads are suspended until a message arrives
    if poll:
        if rank == 0:
            for iRank in range(1, mpi.Get_size()):
                mpi.send(0, iRank)
        else: 
#            print(f'rank {rank} entering polling', file=sys.stderr)
            while True:
                if mpi.iprobe():
                    break
                time.sleep(0.001)
#            print(f'rank {rank} exiting polling', file=sys.stderr)

    i = mpi.bcast(i, root=0) 
    
    if i == sys.maxsize:
        raise SocketClosedException()

    return i

def send_int(conn, i):
    if rank == 0:
        conn.send(i.to_bytes(4, byteorder='little'))

def recv_str(conn):
    length = recv_int(conn)
    data = recv_buffer(conn, length)
    return data.decode('utf-8')
    
def recv_str_mpi(conn):
    s = None
    if rank == 0:
        try:
            s = recv_str(conn)
        except SocketClosedException:
            s = sys.maxsize

    s = mpi.bcast(s, root=0)

    if s == sys.maxsize:
        raise SocketClosedException()

    return s

def recv_vd_mpi(conn, shape = None):
    data = None
    if rank == 0:
        try:
            length = recv_int(conn)
            data = recv_buffer(conn, length)
        except SocketClosedException:
            data = sys.maxsize

    data = mpi.bcast(data, root=0) 

    if data == sys.maxsize:
        raise SocketClosedException()
    
    arr = np.frombuffer(data, dtype=fortran_dt)
    if shape is not None:
        arr = np.reshape(arr, shape, order='F')
    return arr

def send_vd(conn, vd):
    if rank == 0:
        bs = vd.tobytes(order='F')
        send_int(conn, len(bs))
        conn.sendall(bs)


##############################
# MAIN

# Argument parser
parser = argparse.ArgumentParser(description='Script runs a server which is used to predict APTs')
parser.add_argument('--host', type=str, default='/tmp/apt.server.socket', help='The host address to bind to (default: /tmp/apt.server.socket)')
parser.add_argument('--socket_type', type=str, default='unix', help='Socket type to be used "unix" for a local unix socket, "tcp" for a network TCP socket (default: unix)')
parser.add_argument('--port', type=str, default=31415, help='Port number used, only necessary if not a unix socket (default: 31415)')
parser.add_argument('--num_active_processes', type=int, default=0, help='Number of processes which should be active in parallel; can be used to reduce the amount of memory required, by processing only a subset of committee members in parallel; increases computing time, though (Default: 0, full parallelization)')
args = parser.parse_args()


print(os.getpid(), rank, file=sys.stderr)


try:
    conn = None
    if rank == 0:
        # Set up the server:
        if args.socket_type == "unix":
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            
            # if the socket is existing, remove it
            try:
                os.unlink(args.host)
            except OSError:
                if os.path.exists(args.host):
                    raise

            # bind
            server.bind(args.host)

            ready_string = args.host

        else:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind((args.host, args.port))

            ready_string = args.host + ":" + args.port

        # wait for connection
        server.listen()
        print('ready, listening on', ready_string, 'on host', socket.gethostname(), file=sys.stderr)
        this_time = time.perf_counter()
        dtime = this_time - program_start_time
        print(f"initialization took {dtime} seconds", file=sys.stderr)

        # blocking call, wait for 60 seconds for a connection
        server.settimeout(60.)
        try:
            conn, addr = server.accept()
        except socket.timeout:
            print('Waited for 60 seconds, but no incoming connection, exiting...', file=sys.stderr)
            mpi.Abort(10)

        print(f"accepted connection from {addr}", file=sys.stderr)

        server.settimeout(None)

        # timeout for the client of 120s
        conn.settimeout(120.)


    # Message loop
    while True:
        # read an integer
        msg_id = recv_int_mpi(conn, True)

        #print(msg_id, file=sys.stderr)

        if msg_id == -1:
            # this implies that the client has closed the connection
            if rank == 0:
                print('Client has closed the connection, exiting', file=sys.stderr)
            exit(0)

        if rank == 0:
            print('Received Message-ID:', msg_id, file=sys.stderr)
            start_time = time.perf_counter()

        if msg_id == EMessages.PREDICT:
            if rank == 0:
                time_a = datetime.now()

            # read an integer giving how many atoms to expect
            Natoms = recv_int_mpi(conn)
            # receive the symbols:
            symbols_str = recv_str_mpi(conn)
            symbols = symbols_str.split(',')
            # receive the positions:
            # NOTE We are receiving a big Natoms*3 array, however, each vector is sent individuall, thus fortran ordering does not apply here!
            positions = np.reshape(recv_vd_mpi(conn), (Natoms, 3))

            if rank == 0:
                time_b = datetime.now()

            # setup configuration; TODO this conversion should/could be omitted at a later stage!
            atoms = []
            for i in range(Natoms):
                atoms.append(Atom(symbol=symbols[i], position=positions[i]))
            config = Frame(atoms=atoms, box=box)

            # DEBUG
#            with open('tmp.xyz', 'w') as fout:
#                write_conf(fout, config.atoms)

            # do the prediction:
            prediction = net.predict([config])
            if rank == 0:
                time_c = datetime.now()

                pred_apt = prediction['apt']
                pred_var = prediction['std']

                # apply tensor correction  
                summedtensors=np.sum(pred_apt[0],axis=0)
                for i in range(len(pred_apt[0])):
                    pred_apt[0][i] = pred_apt[0][i] - summedtensors/len(pred_apt[0]) 

                # prediction finished, send data
                send_int(conn, EReturnCodes.OK)
                send_vd(conn, pred_apt[0])

                time_d = datetime.now()

                print('Used times in seconds: Recv:', time_b - time_a, 'Predict:', time_c - time_b, 'Send:', time_d - time_c, flush=True)

                # write apt to file?
                if fout_apt is not None or fout_var is not None:
                    for i in range(len(config.atoms)):
                        config.atoms[i].apt = pred_apt[0][i]
                        config.atoms[i].apt_std = pred_var[0][i]

                    if fout_apt is not None:
                        write_conf(fout_apt, config.atoms, meta={'i': timestep},fmt='pa')
                    
                    if fout_var is not None:
                        write_conf(fout_var, config.atoms, meta={'i': timestep}, fmt='ps')

                if fout_frame_uncertainty is not None:
                    max_normal = np.max(prediction['std_norm'][0])
                    mean_normal = np.mean(prediction['std_norm'][0])
                    max_unnormal = np.max(prediction['std'][0])
                    mean_unnormal = np.mean(prediction['std'][0])
                    print(f'{mean_normal} {mean_unnormal} {max_normal} {max_unnormal}', file=fout_frame_uncertainty)

            
            # increment time step
            timestep += 1


        elif msg_id == EMessages.SET:         
            # read another integer informing which variable to set
            var = recv_int_mpi(conn)
            ret = EReturnCodes.OK

            if var == EVariables.APTNN_FILE:
                # read a string
                filename = recv_str_mpi(conn)
                if rank == 0:
                    print('Trying to open', filename, 'as committee model', file=sys.stderr)
                net = CommitteeAPTNN(committee_size=None, model_parameters=None)
                net.load(filename)

                # debug output
                print(f"Rank {rank} finished loading model", file=sys.stderr)


            elif var == EVariables.CELL: 
                cell = recv_vd_mpi(conn, (3,3))
                box.loadFromVectors(cell)

                if rank == 0:
                    print('Received new simulation box definition:', file=sys.stderr)
                    print(cell, file=sys.stderr)


            elif var == EVariables.TIMESTEP: 
                timestep = recv_int_mpi(conn)

            elif var == EVariables.WRITE_APT:
                if rank == 0:
                    # receive filename
                    fn = recv_str(conn)
                    fout_apt = open(fn, 'w')


            elif var == EVariables.WRITE_VAR:
                if rank == 0:
                    # receive filename
                    fn = recv_str(conn)
                    fout_var = open(fn, 'w')

            elif var == EVariables.WRITE_FRAME_UNCERTAINTY:
                if rank == 0:
                    # receive filename
                    fn = recv_str(conn)
                    fout_frame_uncertainty = open(fn, 'w')

                    print('# Per Frame uncertainty', file=fout_frame_uncertainty)
                    print('# Mean (normalized) | Mean (unnormalized) | Max (normalized) | Max (unnormalized)', file=fout_frame_uncertainty)


            # send the return code
            send_int(conn, ret)

        else: 
            # unknown code: 
            if rank == 0:
                print('Received unknown message code!', msg_id, file=sys.stderr)
            exit(1)

        if rank == 0:
            this_time = time.perf_counter()
            dtime = this_time - start_time
            print(f"message processing took {dtime} seconds", file=sys.stderr)
        
except socket.error as err:
    print('Received following socket error, exiting',file=sys.stderr)
    print(err, file=sys.stderr)
    mpi.Abort(1)

except SocketClosedException:
    print(f'rank {rank}: Socket ronnection closed, exiting')
    exit(0)



