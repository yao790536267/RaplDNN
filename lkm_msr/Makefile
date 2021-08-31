# HEADER = -I/usr/include/

ifneq ($(KERNELRELEASE),)
    obj-m := msrdrv.o


else
	KERNELDIR ?= /lib/modules/$(shell uname -r)/build
	PWD := $(shell pwd)

default:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

endif

clean:
	rm -f *.ko
	rm -f *.o
	rm -f *.mod
	rm -f *.mod.c
	rm -f *.symvers
	rm -f *.order
	rm -f *.dwo
	rm -f .*.cmd
	rm -f .*.d