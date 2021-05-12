#!/bin/bash

echo starting camera
cd /home/pi/project
sudo python3 main-code.py

echo Capture again[yes=1 : no=0]
read a
if [$a -eq 0]
then
	sudo python3 main-code.py
else
	echo Attendance take
	
