# dbutils
Utility scripts for db functions

##dbarrival_params.py: Calculates first motion, SNR, maybe polarization from arrival table.

Example to calculate first motion and SNR on P arrivals for a subset of events:
```shell
python dbarrival_params.py -d dbout -o tdoutdb -s 'depth<50. && nass>=10 && lat<=47.25 && lat>=46.75 && lon>=97.75 && lon<=98.25' --dofm --dosnr -a "iphase=~/P/"
```

Example to calculate SNR on S arrivals for a subset of events:
```shell
python dbarrival_params.py -d dbout -o tdoutdb -s 'depth<50. && nass>=10 && lat<=47.25 && lat>=46.75 && lon>=97.75 && lon<=98.25' --dosnr -a "iphase=~/S/"
```
