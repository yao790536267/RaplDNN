import subprocess
import os

# main = "./rapl_tool/sampleRapl -o ./rapl_tool/out.csv"
main = "./rapl_tool/sampleRapl"

## Method 1
# if os.path.exists(main):
#     rc, out = subprocess.getstatusoutput(main)
#     print('rc = %d, \nout : %s' % (rc, out))
#     print("rc= ", rc)
#     print("INSTANCE:  ", isinstance(rc, ))



# print('*' * 100)

## Method 2
f = os.popen(main)
data = f.readlines()
f.close()
print(data)

result = data[0].split(',')
for i in range(len(result)):
    if i == 0:
        print("pkg_total_energy is: ", result[i], ' J')
    elif i == 1:
        print("pkg_current_power is: ", result[i], ' W')
    elif i == 2:
        print("pp0_current_power is: ", result[i], ' W')
    elif i == 3:
        print("pp1_current_power is: ", result[i], ' W')
    elif i == 4:
        print("dram_current_power is: ", result[i], ' W')
    elif i == 5:
        print("total_time is: ", result[i], ' S')
    elif i == 6:
        print("pkg_average_power is: ", result[i], ' W')



# The element order
#        << rapl->pkg_total_energy() << ","
# 		 << rapl->pkg_current_power()<< ","
# 		 << rapl->pp0_current_power()<< ","
# 		 << rapl->pp1_current_power()<< ","
# 		 << rapl->dram_current_power()<< ","
# 		 << rapl->total_time() << ","
# 		 << rapl->pkg_average_power();

# print('*' * 100)
#
# ## Method 3
# os.system(main)