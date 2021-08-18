import pyRAPL

pyRAPL.setup()

# @pyRAPL.measurement(numbers=10)
# def foo():
#     for i in range(10000):
#         x = x + i
#
# for i in range(10000):
#     x = x + i

# meter = pyRAPL.measurement('bar')
# meter.begin()
# x = 1+1
# meter.end()



with pyRAPL.Measurement('bar'):
    x = 1
    for i in range(10000):
        x = x + i

