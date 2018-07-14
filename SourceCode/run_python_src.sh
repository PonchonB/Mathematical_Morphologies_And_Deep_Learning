#!/bin/bash

if [ $# -lt 3 ]
then
	echo "Usage: $0 path_to_file.py path_to_output_file path_to_error_log"
elif [ $# -gt 3 ]
else
	nohup python $1 >$2 2>$3 &
fi
