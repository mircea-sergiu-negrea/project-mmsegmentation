CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# --- BEGIN MOD: creates subdirectory for each new run ---
# Check if user passed --work-dir as an argument; if so, use it as-is.
WORK_DIR_PROVIDED=false
for arg in "${@:3}"; do
    if [[ "$arg" == --work-dir ]]; then
        WORK_DIR_PROVIDED=true
        break
    fi
done

# If --work-dir is not provided, create a timestamped run directory under work_dirs/ap4ad_rgb
if [ "$WORK_DIR_PROVIDED" = false ]; then
    BASE_WORK_DIR="$(dirname $0)/../work_dirs/ap4ad_rgb"  # Parent directory for all runs
    TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)  # Format: YYYY-mm-dd_HH:MM:SS
    RUN_DIR="$BASE_WORK_DIR/run_$TIMESTAMP"  # Full path for this run
    mkdir -p "$RUN_DIR"  # Create the run directory
    # Add --work-dir and the new run dir to the argument list for train.py
    EXTRA_ARGS=(--work-dir "$RUN_DIR" ${@:3})
else
    # If --work-dir is provided, just forward all extra args as-is
    EXTRA_ARGS=(${@:3})
fi
# --- END MOD ---

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch "${EXTRA_ARGS[@]}"  # this passes the created timestamped dir to train.py, so logs go in there
