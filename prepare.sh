#!/bin/bash

S=400

    python3 prepare_data.py --path /media/ferseiti/Elements/Furusato/data/0007/ -s $S -p $S -o /home/ferseiti/packages/upscaling/data/validation/0007_validation.h5 -b 0

    for i in {0008,0011,0016}; do
        python3 prepare_data.py --path /media/ferseiti/Elements/Furusato/data/${i}/ -s $S -p $S -o /home/ferseiti/packages/upscaling/data/train/${i}.h5 -b 0
    done &
