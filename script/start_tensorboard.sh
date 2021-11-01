# Start tensorboard.

if [ "$1" = "" ]; then
  echo "Please specify `logdir`."
  exit
fi

tensorboard --logdir="$1" --bind_all