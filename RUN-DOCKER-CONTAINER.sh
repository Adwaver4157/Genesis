xhost +local:root # コンテナがディスプレイにアクセスできるようにする

docker run --gpus all -it \
    --net host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v $PWD:/workspace \
    --name takanami_genesis \
    genesis
    /bin/bash