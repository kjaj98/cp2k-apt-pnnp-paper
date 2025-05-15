import os
from datetime import datetime
import argparse
import traceback

import numpy as np
import torch
import sys

from mpi4py import MPI

from aptnn.committee import CommitteeAPTNN
from aptnn.box import Box


def load_committee_aptnn(aptnn_file):
    net = CommitteeAPTNN(committee_size=None, model_parameters=None)
    net.load(aptnn_file)
    return net

def load_box(box_file):
    box_file =np.loadtxt(box_file)
    if len(box_file) != 9:
        exit('Requires the periodic lattice in the format "ax ay az bx by bz cx cy cz"')
    box = Box()
    lattice = [[box_file[0], box_file[1], box_file[2]], 
            [box_file[3], box_file[4], box_file[5]],
            [box_file[6], box_file[7], box_file[8]]]
    box.loadFromVectors(lattice)
    return box



