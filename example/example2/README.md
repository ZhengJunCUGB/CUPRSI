This is the case when the input is a file list of binary files.
You can run the program using the following command:

1. si <PassiveSourceData.lst ns=30001 ntr=201 dt=4000 ns_win=10000 outns=1001 StartFldr=1 EndFldr=201 Device=0 threads=256 blocks=128 file_mode=1 RAMSizeMB=20000 VRAMSizeMB=2000 Stack_flag=1 >VirtualShot.su

2. si <PassiveSourceData.lst >VirtualShot.su par=si.par

The file "si.par" includes all the parameters in 1
