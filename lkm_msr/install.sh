#!/bin/bash

if [ "$(whoami)" != "root" ] ; then
        echo -e "\n\tYou must be root to run this script.\n"
        exit 1
fi

make
dmesg -C
#touch /dev/msrdrv
#mknod /dev/msrdrv c 223 0﻿
#chmod 666 /dev/msrdrv
#chmod 666 /dev/cpu/0/msr
#insmod -f msrdrv.ko
#insmod msrdrv.ko

mknod -m 0666 /dev/msrdrv c 223 0 && insmod msrdrv.ko
dmesg
gcc -g -I. -O2 -o sample sample.c

