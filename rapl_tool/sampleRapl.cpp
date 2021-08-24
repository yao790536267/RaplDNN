#include <iostream>
#include <unistd.h>
#include <fstream>
#include <string>

#include "Rapl.h"

using namespace std;


int main(int argc, char *argv[]) {

	// Default settings
	int runtime = 60;         // run for 60 seconds
	int ms_pause = 100;       // sample every 100ms
	bool use_outfile = false; // no output file
	ofstream outfile;

	/**
	 *  Read Commandline Parameters
	 *
	 *  -p -- pause duration (in milliseconds)
	 *  -o -- write output to a file
	 *  -t -- run time (in sec)
	 *
	 *  Example: ./a.out -p 500 -o outfile.csv
	 *
	 **/
	char input_char;
	while ((input_char = getopt(argc, argv, "p:o:")) != -1) {
		switch (input_char) {
		case 'p':
			ms_pause = atoi(optarg);
			break;
		case 't':
			runtime = atoi(optarg);
			break;
		case 'o':
			printf("ouput file:%s\n",optarg);
			use_outfile = true;
			outfile.open(optarg, ios::out | ios::trunc);
			break;
		default:
			abort();
		}
	}

	if (use_outfile) {
		    outfile << "pkg_total_energy J, "
		            << "pkg_current_power, "
					<< "pp0_current_power, "
					<< "pp1_current_power, "
					<< "dram_current_power, "
					<< "total_time" << std::endl;
					}

	Rapl *rapl = new Rapl();

	rapl->sample();

	// Write sample to outfile
	if (use_outfile) {

		outfile << rapl->pkg_total_energy() << ","
			    << rapl->pkg_current_power() << ","
			    << rapl->pp0_current_power() << ","
				<< rapl->pp1_current_power() << ","
				<< rapl->dram_current_power() << ","
				<< rapl->total_time() << endl;
	}
    double reading = rapl->pp0_current_power();
	// Write sample to terminal
//	cout << "\33[2K\r" // clear line
//		 << "\tTotal Energy:" << rapl->pkg_total_energy() << " J"
//		 << "\tpkg Power:" << rapl->pkg_current_power()<< " W"
//		 << "\tpp0 Power:" << rapl->pp0_current_power()<< " W"
//		 << "\tpp1 Power:" << rapl->pp1_current_power()<< " W"
//		 << "\tDram Power:" << rapl->dram_current_power()<< " W"
//		 << "\tTotal Time:" << rapl->total_time() << " sec"
//
////				<< "\tCurrent Time=" << rapl->current_time()<< " sec"
//		 << "\tAverage Power:" << rapl->pkg_average_power() << " W"
//		 << std::endl;

    cout << rapl->pkg_total_energy() << ","
		 << rapl->pkg_current_power()<< ","
		 << rapl->pp0_current_power()<< ","
		 << rapl->pp1_current_power()<< ","
		 << rapl->dram_current_power()<< ","
		 << rapl->total_time() << ","
		 << rapl->pkg_average_power();

		cout.flush();

//	while (rapl->total_time() < runtime) {
//		usleep(1000 * ms_pause);
//		rapl->sample();
//
//		// Write sample to outfile
//		if (use_outfile) {
//
//			outfile << rapl->pkg_total_energy() << ","
//			        << rapl->pkg_current_power() << ","
//					<< rapl->pp0_current_power() << ","
//					<< rapl->pp1_current_power() << ","
//					<< rapl->dram_current_power() << ","
//					<< rapl->total_time() << endl;
//		}
//
//		// Write sample to terminal
//		cout << "\33[2K\r" // clear line
//		        << "\tTotal Energy:" << rapl->pkg_total_energy() << " J"
//		        << "\tpkg Power:" << rapl->pkg_current_power()<< " W"
//				<< "\tpp0 Power:" << rapl->pp0_current_power()<< " W"
//				<< "\tpp1 Power:" << rapl->pp1_current_power()<< " W"
//				<< "\tDram Power:" << rapl->dram_current_power()<< " W"
//				<< "\tTotal Time:" << rapl->total_time() << " sec"
//
////				<< "\tCurrent Time=" << rapl->current_time()<< " sec"
//				<< "\tAverage Power:" << rapl->pkg_average_power() << " W"
//			    << std::endl;
//
//		cout.flush();
//	}


	// Print totals
//	cout << endl
//		<< "\tTotal Energy:\t" << rapl->pkg_total_energy() << " J" << endl
//		<< "\tAverage Power:\t" << rapl->pkg_average_power() << " W" << endl
//		<< "\tTime:\t" << rapl->total_time() << " sec" << endl;

	return 10;
}
