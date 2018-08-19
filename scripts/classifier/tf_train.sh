#!/bin/bash
WIDTH=${TF_WIDTH:-1.0} # Width multipler: 1.0, 0.75, 0.50, 0.25
RES=${TF_RES:-160} # Image resoluton: 224, 192, 160, 128

# Smallest Width=0.25, Res=128
# Biggest Width=1.0, Res=224

#learning_rate: This is something you’ll want to play with. I found 0.0001 to work well.
#testing and validation percentage: The script will split your data into train/val/test for you. 
#   It will use train to train, val to give performance updates every “eval_step_interval”, and 
#   test will run after “how_many_training_steps” to give you your final score.
# validation_batch_size: Setting this to -1 tells the script to use all your data to validate on. 
#   When you don’t have a lot of data (like only 10,000 images), it’s a good idea to use -1 here 
#   to reduce variance between evaluation steps.

OBJ=lamp_light
TRAIN=1

CWD=`pwd`
cd $HOME/programs/tensorflow

# Train model
if [ $TRAIN -eq 1 ]; then
  python tensorflow/examples/image_retraining/retrain.py \
      --image_dir $HOME/Database/objects/$OBJ/$OBJ/ \
      --learning_rate=0.0001 \
      --testing_percentage=20 \
      --validation_percentage=20 \
      --train_batch_size=32 \
      --validation_batch_size=-1 \
      --flip_left_right True \
      --random_scale=30 \
      --random_brightness=30 \
      --eval_step_interval=100 \
      --how_many_training_steps=20000 \
      --output_graph $CWD/../../models/tensorflow/${OBJ}_mn_${WIDTH}_${RES}.pb \
      --architecture mobilenet_${WIDTH}_${RES} > $CWD/../../models/tensorflow/${OBJ}_mn_${WIDTH}_${RES}.log
# Classify model
else 
  python tensorflow/examples/label_image/label_image.py \
    --graph=$CWD/../../models/tensorflow/${OBJ}_mn_${WIDTH}_${RES}.pb \
    --labels=/tmp/output_labels.txt \
    --image=$HOME/Database/objects/$OBJ/$OBJ/on/lamp792.jpg \
    --input_layer=input \
    --output_layer=final_result \
    --input_mean=128 \
    --input_std=128 \
    --input_width=${RES} \
    --input_height=${RES} 
fi   
