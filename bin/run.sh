#!/bin/bash

SOURCE_DIR=$(pwd)/${1:-fastai_category_encoders}
NBS_DIR=$(pwd)/${1:-nbs}

if [[ ! -d $BASE_DIR ]]
then
    SCRIPT_DIR=$(dirname "$(realpath $0)")
    echo < $SCRIPT_DIR/usage.txt
fi

echo "Mounting directories $SOURCE_DIR and $NBS_DIR into the container..."

#  Fork a process to open a new browser tab
sleep 4 && xdg-open http://localhost:8888/lab &



# Run the container, exposing the port
docker run \
    --rm \
    --ipc=host \
    --gpus all \
    -p 8888:8888 -p 8787:8787 -p 8786:8786 -p 9091:22 \
    --mount type=bind,source="$SOURCE_DIR",target=/proj/fastai_category_encoders \
    --mount type=bind,source="$NBS_DIR",target=/proj/nbs \
    fastai_category_encoders