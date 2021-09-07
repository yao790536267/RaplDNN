#!/bin/bash

if [ "$(whoami)" != "root" ] ; then
        echo -e "\n\tYou must be root to run this script.\n"
        exit 1
fi

mknod -m 0666 /dev/msrdrv c 223 0 && insmod msrdrv.ko