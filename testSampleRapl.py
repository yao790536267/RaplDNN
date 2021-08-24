import subprocess
import os

## Method 1
main = "./rapl_tool/sampleRapl"
if os.path.exists(main):
    rc, out = subprocess.getstatusoutput(main)
    print('rc = %d, \nout : %s' % (rc, out))
    print("rc= ", rc)
    print("INSTANCE:  ", isinstance(rc, int))



# print('*' * 100)
#
# ## Method 2
# f = os.popen(main)
# data = f.readlines()
# f.close()
# print(data)
#
# print('*' * 100)
#
# ## Method 3
# os.system(main)