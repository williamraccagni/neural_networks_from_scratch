import scratchnet.layers.linear as ll

class nn:

    def __init__(self):
        self.l1 = ll.linear([[0.2, 0.8, -0.5, 1.0],
                   [0.5, -0.91, 0.26, -0.5],
                   [-0.26, -0.27, 0.17, 0.87]],
                  [2.0, 3.0, 0.5])  # ll1 is the module
        self.l2 = ll.linear([[0.1, -0.14, 0.5],
                   [-0.5, 0.12, -0.33],
                   [-0.44, 0.73, -0.13]],
                  [-1, 2, -0.5])  # ll2 is the module

    def forward(self, x : list):
        return self.l2.forward(self.l1.forward(x))