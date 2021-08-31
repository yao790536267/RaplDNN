#include "msrdrv.h"
#include <linux/module.h>
#include <sys/types.h>
#include <linux/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/errno.h>
#include <sys/ioctl.h>
#include <stdlib.h>
#include <stdio.h>

//#include <c++/9/tr1/stdlib.h>
//#include <c++/9/tr1/stdio.h>

//MODULE_LICENSE("Dual BSD/GPL");

static int loadDriver()
{
    int fd;
//    fd = open("/dev/cpu/0/msr", O_RDWR);
    fd = open("/dev/" DEV_NAME, O_RDWR);
    if (fd == -1) {
        perror("Failed to open /dev/" DEV_NAME);
//        perror("Failed to open /dev/cpu/0/msr");
    }
//    perror("");
    return fd;
}

static void closeDriver(int fd)
{
    int e;
    e = close(fd);
    if (e == -1) {
        perror("Failed to close fd");
    }
}

int main(void)
{
    int fd;
    struct MsrInOut msr_power[] = {
        { MSR_READ, 0x611, 0x00 },       //MSR_PKG_ENERGY_STATUS
        { MSR_READ, 0x639, 0x00 },       //MSR_PP0_ENERGY_STATUS
        { MSR_READ, 0x641, 0x00 },       //MSR_PP1_ENERGY_STATUS
        { MSR_READ, 0x619, 0x00 },       //MSR_DRAM_ENERGY_STATUS
        { MSR_STOP, 0x00, 0x00 }
    };

    fd = loadDriver();
    ioctl(fd, IOCTL_MSR_CMDS, (long long)msr_power);
    printf("test");
    printf("xxx: %7lld\n", msr_power[1].value);

    closeDriver(fd);
    return 0;
}
