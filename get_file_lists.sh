#!/bin/bash

ls train-data/*.tfrecord | shuf > filelist.txt
split -l 64 filelist.txt filelist.
f=filelist.a

cat ${f}a ${f}b ${f}c ${f}d > train-1.list
cat ${f}e > test-1.list

cat ${f}a ${f}b ${f}c ${f}e > train-2.list
cat ${f}d > test-2.list

cat ${f}a ${f}b ${f}e ${f}d > train-3.list
cat ${f}c > test-3.list

cat ${f}a ${f}e ${f}c ${f}d > train-4.list
cat ${f}b > test-4.list

cat ${f}e ${f}b ${f}c ${f}d > train-5.list
cat ${f}a > test-5.list

rm filelist.txt ${f}*
