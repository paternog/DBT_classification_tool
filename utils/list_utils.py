##########################################################################
# Author: Gianfranco Paternò (paterno@fe.infn.it)                        #
# collaboration of INFN Ferrara and INFN Napoli for the next_AIM project #
# Last modification date: March 2024                                     #
#                                                                        #
# SPDX-License-Identifier: EUPL-1.2                                      #
# Copyright 2024 Istituto Nazionale di Fisica Nucleare                   #
##########################################################################


def find_list_max(myList):
    the_max = myList[0]
    imax = 0
    for i in range(1, len(myList)):
        if myList[i] > the_max:
            the_max = myList[i]
            imax = i
    return the_max, imax

def multiply_list_elements(myList):
    result = 1
    for x in myList:
        result = result*x
    return result
    
def multiply_list_elements_to_costant(myList, k):
    newList = []
    for x in myList:
        newList.append(x*k)
    return newList

def repeat_list_elements(myList, k):
    newList = [j for j in myList for i in range(k)]
    return newList   

def list_flatten(myList):
    flat_list = [item for sublist in myList for item in sublist]
    return flat_list

def get_unique_elements(myList):
    res_list = []
    for item in myList: 
        if item not in res_list: 
            res_list.append(item)
    return res_list

def dump_list_to_text_file(fname, mylist):
    file = open(fname,'w')
    for item in mylist:
        file.write(item + "\n")

def remove_multiple_elements_from_list(list1, list2):
    list3 = [item for item in list1 if item not in list2]
    return list3

def copy_multiple_elements_from_list(list1, list2):
    list3 = [item for item in list1 if item in list2]
    return list3
    
