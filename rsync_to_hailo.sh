USER_NAME="katrintomanek"
IP="192.168.0.70"
BASE_DIR="/home/katrintomanek/dev/whisper_on_hailo"
# DRY_RUN="--dry-run"
DRY_RUN=""


#scp hailo_requirements.txt "$USER_NAME@$IP:$BASE_DIR/"

rsync -avz $DRY_RUN -e ssh \
    inference/* "$USER_NAME@$IP:$BASE_DIR/inference"

# rsync -avz $DRY_RUN -e ssh \
#     models/* "$USER_NAME@$IP:$BASE_DIR/models"

# rsync -avz $DRY_RUN -e ssh \
#     samples/* "$USER_NAME@$IP:$BASE_DIR/samples"