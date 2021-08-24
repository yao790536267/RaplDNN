CXX = g++
CXXFLAGS = -std=c++11 -g
LD = $(CXX)
LDFLAGS = -lm -lrt

all : PowerMonitor AppPowerMeter dramOnly sampleRapl

run : PowerMonitor AppPowerMeter dramOnly sampleRapl
	#./PowerMonitor
	./AppPowerMeter sleep 5

sampleRapl : sampleRapl.o Rapl.o
	$(LD) $(LDFLAGS) -o $@ $^

sampleRapl.o : sampleRapl.cpp Rapl.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

dramOnly : dramOnly.o Rapl.o
	$(LD) $(LDFLAGS) -o $@ $^

dramOnly.o : dramOnly.cpp Rapl.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

AppPowerMeter : AppPowerMeter.o Rapl.o
	$(LD) $(LDFLAGS) -o $@ $^

AppPowerMeter.o : AppPowerMeter.cpp Rapl.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

PowerMonitor : PowerMonitor.o Rapl.o
	$(LD) $(LDFLAGS) -o $@ $^

PowerMonitor.o : PowerMonitor.cpp Rapl.h
	$(CXX) $(CXXFLAGS) -c -o $@ $< 

Rapl.o : Rapl.cpp Rapl.h
	$(CXX) $(CXXFLAGS) -c -o $@ $< 

clean :
	rm -f *.o
	rm -f *.csv
	rm -f AppPowerMeter
	rm -f PowerMonitor
	rm -f dramOnly
	rm -f sampleRapl
