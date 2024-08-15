export MODEL_NAME="/home/hyl/hou/DisenBooth-main/stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="/home/hyl/hou/DisenBooth-main/dataset/dog5/"
export OUTPUT_DIR="./output-dog7_multiobject_can"

accelerate launch train_cl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a dog </w> dog next to a can </w> can" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=200 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_prompt="a dog </w> dog next to a can </w> can on the beach" \
  --validation_epochs=200 \
  --seed="0" \
