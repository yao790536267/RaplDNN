import os
import sys
import subprocess

# pid = os.getpid()
# print("py pid：")
# print(pid)
#
# os.system(r'./rapl_tool/AppPowerMeter ' + str(pid))
os.system(r'./rapl_tool/AppPowerMeter sleep 5')

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
