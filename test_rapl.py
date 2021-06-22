import os
import sys
import subprocess

os.system(r'./rapl_tool/AppPowerMeter ' + "sleep 5")

cmd = "./rapl-tool/AppPowerMeter sleep 5"
# subprocess.run(cmd)

# subprocess.Popen("./rapl-tool/AppPowerMeter sleep 5")

# main = "./rapl-tool/AppPowerMeter sleep 5"
# if os.path.exists(main):
#     subprocess.getstatusoutput(main)

sys.exit(0)
