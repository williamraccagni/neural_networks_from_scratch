# IMPORTS
import scratchnet.nn as nn
import scratchnet.layers.linear as ll
from nnfs.datasets import spiral_data

if __name__ == '__main__':

    l1 = ll.linear([[0.2, 0.8, -0.5, 1.0],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]],
                   [2.0, 3.0, 0.5]) # ll is the module
    
    print(l1.forward([[1.0, 2.0, 3.0, 2.5],
                      [2.0, 5.0, -1.0, 2.0],
                      [-1.5, 2.7, 3.3, -0.8]]))
    
    # l2 = ll.linear(3, 4)
    # print(l2.forward([[1.0, 2.0, 3.0, 2.5],
    #                   [2.0, 5.0, -1.0, 2.0],
    #                   [-1.5, 2.7, 3.3, -0.8]]))
    
    
    n1 = nn.nn()
    print(n1.forward([[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]]))