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
#include <string.h>
#include <sys/shm.h>
#include <string.h>

//#include <c++/9/tr1/stdlib.h>
//#include <c++/9/tr1/stdio.h>

//MODULE_LICENSE("Dual BSD/GPL");

typedef unsigned int	uint;
typedef unsigned long	ulong;

//#define DEV_NAME	"rdmsr"

#define IOCTL_MAGIC	31337
#define IOCTL_MSR_READ _IOR(IOCTL_MAGIC, 0, ulong*)

static int loadDriver()
{
    int fd;
//    printf("before open");
//    fd = open("/dev/cpu/0/msr", O_RDWR);
    fd = open("/dev/" DEV_NAME, O_RDWR);
//    printf("after open");
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

static int ioctl_rdmsr(int f_drv, ulong f_msr, ulong* f_pVal)
{
	ulong val = f_msr;
	int err = ioctl(f_drv, IOCTL_MSR_READ, &val);
	if(err < 0)
	{
		printf("ERROR: failed to send IOCTL_MSR_READ (%d)\n", err);
		return -ENOTTY;
	}

	*f_pVal = val;
	return 0;
}

static int shared_memory_alloc(int pkg, int pp0, int pp1, int dram)
{

//    printf("shared_memory_alloc pkg: %d\n", pkg);       //MSR_PKG_ENERGY_STATUS
//    printf("shared_memory_alloc pp0: %d\n", pp0);       //MSR_PP0_ENERGY_STATUS
//    printf("shared_memory_alloc pp1: %d\n", pp1);       //MSR_PP1_ENERGY_STATUS
//    printf("shared_memory_alloc dram: %d\n", dram);       //MSR_DRAM_ENERGY_STATUS

    struct shared_powers
    {
        int pkg;
        int pp0;
        int pp1;
        int dram;
    };
    void *shm = NULL;
//    struct shared_powers *shared;
    int shmid;
    int strSize = 128;

//    printf("size of struct: %d", (int)sizeof(struct shared_powers)); //16

//    int powers[4];
//    printf("size of int[4]: %d", (int)sizeof(int [4])); //int[4]:16 ; long long[4] = 32
//    shmid = shmget( (key_t)1234, sizeof(int [4]), 0666|IPC_CREAT);

//    shmid = shmget( (key_t)1234, sizeof(struct shared_powers), 0666|IPC_CREAT);
    shmid = shmget( (key_t)1234, 1024, 0666|IPC_CREAT);

    if (shmid == -1) {
        printf("shmget failed\n");
        return -1;
    }

    shm = shmat(shmid, 0, 0);
    if (shm == (void *)-1) {
        printf("shmat failed\n");
        return -1;
    }
//    printf("\nShared memory attached\n");

    char* shared;
    shared = (char*)shm;
    char* str1 = "{\"pkg\":";
    char* str2 = ",\"pp0\":";
    char* str3 = ",\"pp1\":";
    char* str4 = ",\"dram\":";
    char* str5 = "}";
//    int jsonSize = strlen(str1) + strlen(str2) + strlen(str3) + strlen(str4) + strlen(str5) + strSize;
//    int jsonSize;
//    jsonSize = 1024;
//    printf("jsonSize : %d\n", jsonSize);
//    char *name = (char *) malloc(jsonSize+1);
    char jsonStr[1024];
//    printf("jsonStr 11\n");
    sprintf(jsonStr, "%s%d%s%d%s%d%s%d%s", str1, pkg, str2, pp0, str3, pp1, str4, dram, str5);

//    printf("jsonStr : %s\n\n", jsonStr);
    strcpy(shared, jsonStr);
//    printf("value from shared memory:  %s\n", shared);
//    printf("shared 22\n");
//    shared = (struct shared_powers*)shm;
//    shared->pkg = pkg;
//    shared->pp0 = pp0;
//    shared->pp1 = pp1;
//    shared->dram = dram;

//    powers = shm;
//    powers[0] = pkg;
//    powers[1] = pp0;
//    powers[2] = pp1;
//    powers[3] = dram;

    if (shmdt(shm) == -1)
    {
        printf("shmdt failed\n");
        return -1;
    }
    return 0;
}

int main(int argc, char** argv)
{
//    if(argc != 2)
//	{
//		printf("Usage: <rdmsr> msr_number\n");
//		return 0;
//	}

//    char* str = argv[1];
//    printf("argv 1: %s\n", str);

//	char* errstr = NULL;
//	ulong msr = strtol(argv[1], &errstr, 0);
//	if(*errstr)
//	{
//		printf("ERROR: invalid MSR\n");
//		return -EINVAL;
//	}

//	int file = open("/dev/"DEV_NAME, O_RDWR);
//	if(file < 0)
//	{
//		printf("ERROR: failed to open the device \"/dev/"DEV_NAME"\"\n");
//		return -EBADF;
//	}

//	ulong val;
//	if(ioctl_rdmsr(file, msr, &val) == 0)
//		printf("MSR 0x%08X = 0x%016lX\n", (uint)msr, val);

//	close(file);


    int fd;
//    printf("before MsrInOut");
    struct MsrInOut msr_power[] = {
        { MSR_READ, 0x611, 0x00 },       //MSR_PKG_ENERGY_STATUS
        { MSR_READ, 0x639, 0x00 },       //MSR_PP0_ENERGY_STATUS
        { MSR_READ, 0x641, 0x00 },       //MSR_PP1_ENERGY_STATUS
        { MSR_READ, 0x619, 0x00 },       //MSR_DRAM_ENERGY_STATUS
        { MSR_STOP, 0x00, 0x00 }
    };
//    printf("before loadDriver");
    fd = loadDriver();
//    printf("before ioctl");
    ioctl(fd, IOCTL_MSR_CMDS, (long long)msr_power);
//    printf("after ioctl");1

//    printf("MsrInOut Values: \n");
//    printf("MSR_PKG_ENERGY_STATUS: %7lld\n", msr_power[0].value);       //MSR_PKG_ENERGY_STATUS
//    printf("MSR_PP0_ENERGY_STATUS: %7lld\n", msr_power[1].value);       //MSR_PP0_ENERGY_STATUS
//    printf("MSR_PP1_ENERGY_STATUS: %7lld\n", msr_power[2].value);       //MSR_PP1_ENERGY_STATUS
//    printf("MSR_DRAM_ENERGY_STATUS: %7lld\n", msr_power[3].value);       //MSR_DRAM_ENERGY_STATUS

//    int result = 0;
//
//    if (strcmp(str, "pkg") == 0 ) {
//        result = msr_power[0].value;
//    }
//    else if (strcmp(str, "pp0") == 0 ) {
//        result = msr_power[1].value;
//    }
//    else if (strcmp(str, "pp1") == 0 ) {
//        result = msr_power[2].value;
//    }
//    else if (strcmp(str, "dram") == 0 ) {
//        result = msr_power[3].value;
//    }


//    switch (str) {
//        case "pkg":
//            result = msr_power[0].value;
//            break;
//        case "pp0":
//            result = msr_power[1].value;
//            break;
//        case "pp1":
//            result = msr_power[2].value;
//            break;
//        case "dram":
//            result = msr_power[3].value;
//            break;
//        default:
//            result = 0;
//    }
//    printf("xxxxx result: %7lld\n", result);

    closeDriver(fd);

    if (shared_memory_alloc((int)msr_power[0].value, (int)msr_power[1].value, (int)msr_power[2].value, (int)msr_power[3].value) == -1)
    {
        printf("\ncall shared memory alloc wrong\n");
    }

    return 0;
}
