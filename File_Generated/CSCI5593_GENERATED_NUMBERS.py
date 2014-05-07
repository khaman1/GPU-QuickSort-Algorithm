import numpy
import random

filename = open('numbers.txt', 'w')

NUMBERS = 100000
ARRAY = numpy.empty(NUMBERS)

filename.write(str(NUMBERS) + '\n')


for i in range (0,NUMBERS):
    new_line = str(random.randint(0,1000)) + '\n'
    filename.write(new_line)

filename.close()
