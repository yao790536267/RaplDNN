import os
import sys
import subprocess
import time
import rapl

# pid = os.getpid()
# print("py pid：")
# print(pid)
#
# os.system(r'./rapl_tool/AppPowerMeter ' + str(pid))
# os.system(r'./rapl_tool/AppPowerMeter sleep 5')

# cmd = "./rapl-tool/AppPowerMeter sleep 5"
# subprocess.run(cmd)

# subprocess.Popen("./rapl-tool/AppPowerMeter sleep 5")

# main = "./rapl-tool/AppPowerMeter sleep 5"
# if os.path.exists(main):
#     subprocess.getstatusoutput(main)
import time

# org_pid = os.getpid()
# child_pid = os.fork()
#
# print("11 py self pid in parent", org_pid)
#
# if child_pid >= 0:
#     if child_pid == 0:
#         # current_pid = os.getpid()
#         print('ttttxxxx')
#         # print("py pid in child: ", current_pid)
#
#         os.system(r'./rapl_tool/AppPowerMeter ' + str(org_pid))
#         print("cpp endgs")
#
#     else:
#         print("22 py child_pid in parent: ", child_pid)
#         print("33 py self pid in parent", os.getpid())
#         time.sleep(5)
#
# else:
#     print("Python fork fail")
#
# sys.exit(0)

s1 = rapl.RAPLMonitor.sample()
time.sleep(5)
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

#print "S1", s1.energy("package-0", "core", rapl.WATT_HOURS)
#s1.dump()

#print "S2", s2.energy("package-0", "core", rapl.WATT_HOURS)
#s2.dump()