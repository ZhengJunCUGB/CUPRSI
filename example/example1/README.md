This is the case when the input is a single binary file.
You can run the program using the following command:

1. si <in.bin ns=30001 ntr=201 dt=4000 ns_win=5000 outns=1001 StartFldr=1 EndFldr=201 Device=0 threads=256 blocks=128 file_mode=0 RAMSizeMB=20000 VRAMSizeMB=2000 Stack_flag=1 >out.su

2. si <in.bin >out.su par=si.par

The file "si.par" includes all the parameters in 1
