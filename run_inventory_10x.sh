#!/bin/sh
set -e

for i in $(seq 1 10); do
  echo "Run $i/10, Testcase 4"
  python inventory_single_2.py --species-file testcase4.json --log-run
done
