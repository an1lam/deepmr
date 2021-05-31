EC2_DNS=${1:-"ec2-34-203-38-98.compute-1.amazonaws.com	"}
PEM_FILE_PATH=${2:-"~/StephenDAK_Dell.pem"}
PORT=${3:-8888}
ssh -i $PEM_FILE_PATH ubuntu@$EC2_DNS -L $PORT:localhost:$PORT
