#!/bin/bash

echo "Just rolling with the alternate"
echo 'PREPARING TRAINING DATA...'
python make_vctk.py --force-gen --cfg "cfg/e2e_data.cfg" --out_file "vctk.tfrecords" --num_slice 6 --sample_rate 8192