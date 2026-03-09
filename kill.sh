#!/bin/bash
PID=$(pgrep -u zhaoxiudi -f "(python|python3) testdata.py")
if [ -n "$PID" ]; then
  kill -9 $PID
fi
