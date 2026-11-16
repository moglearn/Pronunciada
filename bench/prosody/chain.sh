#!/bin/sh
# wait for PID $1 to exit, then run the rest of the arguments as a command
pid=$1; shift
while kill -0 "$pid" 2>/dev/null; do sleep 10; done
exec "$@"
