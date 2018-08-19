#!/bin/bash
trap ctrl_c INT
function ctrl_c() {
  pkill -P $$ --signal 9
	pkill python
  exit
}
#export TF_WIDTH=1.0
#export TF_RES=160
#./tf_train.sh
#sleep 2

#export TF_WIDTH=0.25
#export TF_RES=128
#./tf_train.sh
#sleep 2

#export TF_WIDTH=0.50
#export TF_RES=192
#./tf_train.sh
#sleep 2

#export TF_WIDTH=1.0
#export TF_RES=224
#./tf_train.sh
#sleep 2

#python keras_state_trainer.py -c light,lamp -i 0 -s 75
#python keras_state_trainer.py -c light,lamp -i 1 -s 75
#python keras_state_trainer.py -c light,lamp -i 2 -s 75
#python keras_state_trainer.py -c light,lamp -i 0 -s 128
#python keras_state_trainer.py -c light,lamp -i 1 -s 128 
#python keras_state_trainer.py -c light,lamp -i 2 -s 128
for i in {0..50}; do
  python cnn_trainer.py -c light,lamp -ch 3 -d uint8
  python cnn_trainer.py -c tv -ch 3 -d uint8
  python cnn_trainer.py -c head -ch 1 -d uint16 -b 1.5
done

wait
