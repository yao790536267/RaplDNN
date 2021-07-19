#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <string.h>
#include <iostream>
#include <fstream>

#include "Rapl.h"

int main(int argc, char *argv[]) {

	Rapl *rapl = new Rapl();

	std::cout << std::endl
				<< "\tTotal Energy:\t" << rapl->pkg_total_energy() << " J" << std::endl
				<< "Current Power=" << rapl->pkg_current_power() << std::endl
				<< "\tCurrent Time=" << rapl->current_time() << std::endl
				<< "\tAverage Power:\t" << rapl->pkg_average_power() << " W" << std::endl
				<< "\tTotal Time:\t" << rapl->total_time() << " sec" << std::endl
				<< "\tpkg Power:\t" << rapl->pkg_current_power() << std::endl
				<< "\tpp0 Power:\t" << rapl->pp0_current_power() << std::endl
				<< "\tpp1 Power:\t" << rapl->pp1_current_power() << std::endl
				<< "\tDram Power:\t" << rapl->dram_current_power() << std::endl;

	float dram = rapl->dram_current_power();
	return 0;
}
