import numpy as np  


class fibonacciClass():
    def __init__(self):
        
        self.numbers_list = []
    def __iter__(self):
        return self

    def __next__(self):
        return_value = self.numbers_list
        while 1:
            if(len(self.numbers_list) < 2):
                self.numbers_list.append(1)
            else:
                self.numbers_list.append(self.numbers_list[-1] + self.numbers_list[-2])
            yield return_value # change this line so it yields its list instead of 1

our_generator = fibonacciClass()
my_output = []

for i in range(10):
    print(next(our_generator ) )
        