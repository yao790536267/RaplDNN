#!/bin/bash

if [ "$(whoami)" != "root" ] ; then
        echo -e "\n\tYou must be root to run this script.\n"
        exit 1
fi
#chmod 666 /dev/msrdrv
chmod 666 /dev/cpu/0/msr

dmesg -C
#insmod -f msrdrv.ko
insmod msrdrv.ko
rmmod msrdrv
dmesg -c

#insmod sample
#rmmod sample
#dmesg -c