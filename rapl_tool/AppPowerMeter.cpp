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

//     std::cout<<"argv【0】 "<<argv[0]<< std::endl;
//     std::cout<<"argv【1】 "<<argv[1]<< std::endl;
//     std::cout<<"int argv[1] "<<std::__cxx11::stoi(argv[1])<< std::endl;
//     std::cout<<"argv【2】 "<<argv[2]<< std::endl;

	Rapl *rapl = new Rapl();
	int ms_pause = 100;       // sample every 100ms
	std::ofstream outfile ("./rapl_DATA.csv", std::ios::out | std::ios::trunc);

//    int pid = std::__cxx11::stoi(argv[1]);
//    pid_t child_pid = pid;
	pid_t child_pid = fork();
//	std::cout<<"child_pid : "<<child_pid<<std::endl;
//	std::cout<<"Type of child_pid : "<<typeid(child_pid).name()<<std::endl;
	int cpp_pid = getpid();
	std::cout<<"CPP PID : "<<cpp_pid<<std::endl;

	if (child_pid >= 0) { //fork successful
		if (child_pid == 0) { // child process

			// printf("CHILD: child pid=%d\n", getpid());

            // execute the application
			int exec_status = execvp(argv[1], argv+1);

			if (exec_status) {
				std::cerr << "execv failed with error " 
					<< errno << " "
					<< strerror(errno) << std::endl;
			}

		} else {              // parent process

		    outfile << "pkg_current_power, "
					<< "pp0_current_power, "
					<< "pp1_current_power, "
					<< "dram_current_power, "
					<< "total_time" << std::endl;
			
			int status = 1;
			// child_pid = ?
			waitpid(child_pid, &status, WNOHANG);
			while (status) {
				
				usleep(ms_pause * 1000);

				// rapl sample
				rapl->sample();
				outfile << rapl->pkg_current_power() << ","
					<< rapl->pp0_current_power() << ","
					<< rapl->pp1_current_power() << ","
					<< rapl->dram_current_power() << ","
					<< rapl->total_time() << std::endl;

				waitpid(child_pid, &status, WNOHANG);	
			}
			wait(&status); /* wait for child to exit, and store its status */
			std::cout << "PARENT: Child's exit code is: " 
				<< WEXITSTATUS(status) 
				<< std::endl;
			
			std::cout << std::endl 
				<< "\tTotal Energy:\t" << rapl->pkg_total_energy() << " J" << std::endl
				<< "\tAverage Power:\t" << rapl->pkg_average_power() << " W" << std::endl
				<< "\tTime:\t" << rapl->total_time() << " sec" << std::endl
				<< "CPP end !!";
		}

	} else {
	// FAIL TO FORK
		std::cerr << "fork failed" << std::endl;
		return 1;
	}
	
	return 0;
}
