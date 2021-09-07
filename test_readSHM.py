import re
from ctypes import *
import numpy as npimport

SHM_SIZE = 1024
SHM_KEY = 1234

try:
    rt = CDLL('librt.so')
except:
    rt = CDLL('librt.so.1')
shmget = rt.shmget
shmget.argtypes = [c_int, c_size_t, c_int]
shmget.restype = c_int
shmat = rt.shmat
shmat.argtypes = [c_int, POINTER(c_void_p), c_int]
shmat.restype = c_void_p

shmid = shmget(SHM_KEY, SHM_SIZE, 0o666)

if shmid < 0:
    print("shmid < 0")
else:

    addr = shmat(shmid, None, 0)
    jsonStr = string_at(addr, SHM_SIZE)

    jsonStr = jsonStr.decode()
    jsonStr = re.sub('\\x00', "", jsonStr)
    print("jsonStr: ", jsonStr)
    infoStr = jsonStr
    # infoStr = jsonStr.strip(b"\x00".decode())
    # print(infoStr, type(infoStr))

    import json
    info = json.loads(infoStr)
    # print(info, type(info))
    print("info pkg: ", info["pkg"])
    print("info pp0: ", info["pp0"])
    print("info pp1: ", info["pp1"])
    print("info dram: ", info["dram"])

    # f = open(OUTFILE, 'wb')
    # rate = int.from_bytes(string_at(addr, 4), byteorder='little', signed=True)  # 这里数据文件是小端int16类型
    # len_a = int.from_bytes(string_at(addr + 4, 4), byteorder='little', signed=True)
    # len_b = int.from_bytes(string_at(addr + 8, 4), byteorder='little', signed=True)
    # print(rate, len_a, len_b)
    # f.write(string_at(addr + 12, SHM_SIZE))
    # f.close()
# print ("Dumped %d bytes in %s" % (SHM_SIZE, OUTFILE))
# print("Success!", datetime.datetime.now() - begin_time)