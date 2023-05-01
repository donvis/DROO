import numpy as np
import math

arr = np.array([1, 2, 3, 4])
arr2 = np.array([0, 0, 0])

def computation_rate():
def allocate_min(arr1, arr2):
    # for the very first iteration, since all servers have no load,
    # allocate resources to first server
    arr2[0] += arr1[0]
    arr1 = np.delete(arr1, [0])
    # Do this until we iterate through all
    while len(arr1) != 0:
        tempvarlist = []
        for i in range(len(arr2)):
            templist = [np.copy(arr2), np.copy(arr2), np.copy(arr2)]
            templist[i][i] += arr1[0]
            tempvarlist.append(templist[i])
        for i in range(len(tempvarlist)):
            templist[i] = np.var(tempvarlist[i])
            print(templist)
        min_var_element = min(templist)
        min_array = np.where(templist == min_var_element)[0][0]
        print(templist)
        print(min_array)
        arr2 = np.copy(tempvarlist[min_array])
        print(arr2)
        arr1 = np.delete(arr1, [0])


def allocate_max(arr1, arr2):
    # for the very first iteration, since all servers have no load,
    # allocate resources to first server
    arr2[0] += arr1[0]
    arr1 = np.delete(arr1, [0])
    # Do this until we iterate through all
    while len(arr1) != 0:
        tempvarlist = []
        for i in range(len(arr2)):
            templist = [np.copy(arr2), np.copy(arr2), np.copy(arr2)]
            templist[i][i] += arr1[0]
            tempvarlist.append(templist[i])
        for i in range(len(tempvarlist)):
            templist[i] = np.var(tempvarlist[i])
            print(templist)
        max_var_element = max(templist)
        max_array = np.where(templist == max_var_element)[0][0]
        print(templist)
        print(max_array)
        arr2 = np.copy(tempvarlist[max_array])
        print(arr2)
        arr1 = np.delete(arr1, [0])

    # for i in range(len(arr1)):
    # tempvar = np.var(arr2[])

    # tempvar
    return


allocate_min(arr, arr2)
allocate_max(arr, arr2)
# print(np.var(np.array([1,0,2])))