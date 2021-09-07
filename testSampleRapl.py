import subprocess
import os

# main = "./rapl_tool/sampleRapl -o ./rapl_tool/out.csv"
# main = "./rapl_tool/sampleRapl"
main = "./lkm_msr/sample"

# make_term = "sudo sh ./make.sh"
# mknod_term = "sudo sh ./mknod.sh"
# uninstall_term = "sudo sh ./uninstall.sh"
# sample_term = "./sample"

# os.system("cd lkm_msr/")
# password = '0808'
# os.system('echo %s|sudo -S %s' % (password, "sudo su"))
# os.system('echo {}|sudo -S {}'.format(password, make_term))
# os.system('echo {}|sudo -S {}'.format('0808', mknod_term))
# os.system('echo {}|sudo -S {}'.format('0808', sample_term))
# os.system('echo {}|sudo -S {}'.format('0808', uninstall_term))
# os.system(make_term)
# os.system(mknod_term)
# os.system(sample_term)
# os.system(uninstall_term)

## Method 1

# print('\n\n******* getstatusouput METHOD: ')
# if os.path.exists(main):
#     return_value, out = subprocess.getstatusoutput(main)
#     print('return_value = %d, \nout : %s' % (return_value, out))
#     # print("rc= ", rc)
#     # print("INSTANCE:  ", isinstance(rc, int))
#
# print("\ngetstatusouput return value : ", return_value)


# print('*' * 100)

## Method 2
# f = os.popen(main)
# data = f.readlines()
# f.close()
# print(data)
#
# result = data[0].split(',')
# for i in range(len(result)):
#     if i == 0:
#         print("pkg_total_energy is: ", result[i], ' J')
#     elif i == 1:
#         print("pkg_current_power is: ", result[i], ' W')
#     elif i == 2:
#         print("pp0_current_power is: ", result[i], ' W')
#     elif i == 3:
#         print("pp1_current_power is: ", result[i], ' W')
#     elif i == 4:
#         print("dram_current_power is: ", result[i], ' W')
#     elif i == 5:
#         print("total_time is: ", result[i], ' S')
#     elif i == 6:
#         print("pkg_average_power is: ", result[i], ' W')



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
# print('\n\n******* os.system METHOD: ')
# # os.system(main)
# return_value = os.system(main)
# print("\nos system return value : ", return_value)