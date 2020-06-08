export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

MODEL="MyModel" #or MYCARLAB_STUDIO

TRAIN_BATCH_SIZE=1
TRAIN_CROP_SIZE=1000
TRAIN_VAL_RATIO=0.9
NUM_OF_CLASSES=2

NUM_ITERATIONS=20000

STORAGE_DIR="DATA/${MODEL}"
INPUT_DIR="${STORAGE_DIR}/Raw"
OUTPUT_DIR="${STORAGE_DIR}/Dataset"

MODEL_VARIANT="xception_71"

INITCHECKPOINT_PATH="${STORAGE_DIR}/Model/init_model/model.ckpt"
TRAINLOG_DIR="${STORAGE_DIR}/Model/train_log"
CKPT_PATH="${TRAINLOG_DIR}/model.ckpt-${NUM_ITERATIONS}"
FROZEN_GRAPH_PATH="${STORAGE_DIR}/Model/frozen_graph/frozen_inference_graph.pb"


DATASET_DIR="${OUTPUT_DIR}/tfrecord"

python3 build_dataset.py \
    --input_path=${INPUT_DIR} \
    --output_path=${OUTPUT_DIR} \
    --train_crop_size=${TRAIN_CROP_SIZE} \
    --train_val_ratio=${TRAIN_VAL_RATIO}

python3 deeplab/train.py \
    --logtostderr \
    --model_variant=${MODEL_VARIANT} \
    --dense_prediction_cell_json="deeplab/core/dense_prediction_cell_branch5_top1_cityscapes.json" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_batch_size=${TRAIN_BATCH_SIZE} \
    --training_number_of_steps="${NUM_ITERATIONS}" \
    --fine_tune_batch_norm=false \
    --initialize_last_layer=false \
    --last_layers_contain_logits_only=true \
    --tf_initial_checkpoint="${INITCHECKPOINT_PATH}" \
    --train_logdir="${TRAINLOG_DIR}" \
    --dataset_dir="${DATASET_DIR}" \
    --num_of_classes=${NUM_OF_CLASSES} \
    --train_crop_size="${TRAIN_CROP_SIZE},${TRAIN_CROP_SIZE}" \
    --base_learning_rate=0.0001 \

python3 deeplab/eval.py \
    --logtostderr \
    --model_variant=${MODEL_VARIANT} \
    --dense_prediction_cell_json="deeplab/core/dense_prediction_cell_branch5_top1_cityscapes.json" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_logdir="${EVALLOG_DIR}" \
    --dataset_dir="${DATASET_DIR}" \
    --checkpoint_dir="${TRAINLOG_DIR}" \
    --eval_crop_size="${TRAIN_CROP_SIZE},${TRAIN_CROP_SIZE}" \
    --max_number_of_evaluations=1 

python3 deeplab/export_model.py \
    --checkpoint_path=${CKPT_PATH} \
    --export_path=${FROZEN_GRAPH_PATH} \
    --model_variant=${MODEL_VARIANT} \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --num_classes=${NUM_OF_CLASSES} \
    --crop_size=${TRAIN_CROP_SIZE} \
    --crop_size=${TRAIN_CROP_SIZE} \
    --inference_scales=1.0 \
    --dense_prediction_cell_json="deeplab/core/dense_prediction_cell_branch5_top1_cityscapes.json" \
