import rapl
import pyRAPL
import math

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
# x = 1
# for i in range(100000000):
#     x = x + i
# meter.end()

TEST_COUNT = 10

for i in range(1):
    with pyRAPL.Measurement('bar'):
        print("Test pyRAPL begins")
        x = 1
        for i in range(TEST_COUNT):
            # x = x + i
            # x = math.pow(x, 2)
            x = x * x

    # rapl
    s1 = rapl.RAPLMonitor.sample()

    print("Test rapl begins")
    x = 1
    for i in range(TEST_COUNT):
        x = x * x
        # x = math.pow(x, 2)

    s2 = rapl.RAPLMonitor.sample()
    diff = s2 - s1
    for d in diff.domains:
        domain = diff.domains[d]
        power = diff.average_power(package=domain.name)
        print("%s = %0.2f W" % (domain.name, power))

        for sd in domain.subdomains:
            subdomain = domain.subdomains[sd]
            power = diff.average_power(package=domain.name, domain=subdomain.name)
            print("\t%s = %0.2f W" % (subdomain.name, power))




