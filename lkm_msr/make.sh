#!/bin/bash

make
gcc -g -I. -O2 -o sample sample.c

