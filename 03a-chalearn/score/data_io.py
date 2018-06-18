#!/usr/bin/env python

# Data IO library
# Isabelle Guyon, ChaLearn, March-September 2014

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

import csv
import numpy as np
import os
import pandas as pd
import pickle
import shutil

from sys import stderr
from sys import version
from glob import glob as ls
from os import getcwd as pwd
from pip import get_installed_distributions as lib

import yaml

if (os.name == "nt"):
    filesep = '\\'
else:
    filesep = '/'

swrite = stderr.write


def write_list(lst):
    for item in lst:
        swrite(item + "\n")

    # Create a directory if it does not exist


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def rmdir(d):
    if os.path.exists(d):
        shutil.rmtree(d)


def inventory_data_nodir(input_dir):
    # THIS IS THE OLD STYLE WITH NO SUB-DIRECTORIES
    training_names = ls(os.path.join(input_dir, '*_train.data'))
    for i in range(0, len(training_names)):
        name = training_names[i]
        training_names[i] = name[-name[::-1].index(filesep):-name[::-1].index('_') - 1]
        check_dataset(input_dir, training_names[i])
    return training_names


def check_dataset(dirname, name):
    # Check the test and valid files are there
    valid_file = os.path.join(dirname, name + '_valid.data')
    if not os.path.isfile(valid_file):
        print('No validation file for ' + name)
        exit(1)
    test_file = os.path.join(dirname, name + '_test.data')
    if not os.path.isfile(test_file):
        print('No test file for ' + name)
        exit(1)
    # Check the training labels are there
    training_solution = os.path.join(dirname, name + '_train.solution')
    if not os.path.isfile(training_solution):
        print('No training labels for ' + name)
        exit(1)
    return True


def inventory_data(input_dir):
    training_names = ls(input_dir + '/*/*_train.data')
    for i in range(0, len(training_names)):
        name = training_names[i]
        training_names[i] = name[-name[::-1].index(filesep):-name[::-1].index('_') - 1]
        check_dataset(os.path.join(input_dir, training_names[i]), training_names[i])
    return training_names


# def data(filename):
#    return pd.read_csv(filename, sep=' ')

def data(filename):
    return np.genfromtxt(filename)


def write(filename, predictions):
    with open(filename, "w") as output_file:
        for val in predictions:
            output_file.write('{:5.4g}\n'.format(val))


def show_io(input_dir, output_dir):
    swrite('\n=== DIRECTORIES ===\n\n')
    # Show this directory
    swrite("-- Current directory " + pwd() + ":\n")
    write_list(ls('.'))
    write_list(ls('./*'))
    write_list(ls('./*/*'))
    swrite("\n")

    # List input and output directories
    swrite("-- Input directory " + input_dir + ":\n")
    write_list(ls(input_dir))
    write_list(ls(input_dir + '/*'))
    write_list(ls(input_dir + '/*/*'))
    write_list(ls(input_dir + '/*/*/*'))
    swrite("\n")
    swrite("-- Output directory  " + output_dir + ":\n")
    write_list(ls(output_dir))
    write_list(ls(output_dir + '/*'))
    swrite("\n")

    # write meta data to sdterr
    swrite('\n=== METADATA ===\n\n')
    swrite("-- Current directory " + pwd() + ":\n")
    try:
        metadata = yaml.load(open('metadata', 'r'))
        for key, value in metadata.items():
            swrite(key + ': ')
            swrite(str(value) + '\n')
    except:
        swrite("none\n");
    swrite("-- Input directory " + input_dir + ":\n")
    try:
        metadata = yaml.load(open(os.path.join(input_dir, 'metadata'), 'r'))
        for key, value in metadata.items():
            swrite(key + ': ')
            swrite(str(value) + '\n')
        swrite("\n")
    except:
        swrite("none\n");


def show_version():
    # Python version and library versions
    swrite('\n=== VERSIONS ===\n\n')
    # Python version
    swrite("Python version: " + version + "\n\n")
    # Give information on the version installed
    swrite("Versions of libraries installed:\n")
    map(swrite, sorted(["%s==%s\n" % (i.key, i.version) for i in lib()]))
