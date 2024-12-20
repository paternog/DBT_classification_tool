##########################################################################
# Author: Gianfranco Paternò (paterno@fe.infn.it)                        #
# collaboration of INFN Ferrara and INFN Napoli for the next_AIM project #
# Last modification date: March 2024                                     #
#                                                                        #
# SPDX-License-Identifier: EUPL-1.2                                      #
# Copyright 2024 Istituto Nazionale di Fisica Nucleare                   #
##########################################################################


import subprocess
import os


def get_gpu_memory_usage(gpu_id):
    """
    function to check the gpu memory usage
    """
    command = f"nvidia-smi --id={gpu_id} --query-gpu=memory.used --format=csv"
    output_cmd = subprocess.check_output(command.split())    
    memory_used = output_cmd.decode("ascii").split("\n")[1]
    # Get only the memory part as the result comes as '10 MiB'
    memory_used = int(memory_used.split()[0])
    return memory_used


def get_free_gpu_memory(gpu_id):
    """
    function to get the free GPU memory in MegaBytes for each GPU
    """
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values[gpu_id]
