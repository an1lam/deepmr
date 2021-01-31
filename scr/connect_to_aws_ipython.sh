PEM_FILE_PATH=${1:-"~/StephenDAK_Dell.pem"}
EC2_DNS=${2:-"ec2-34-203-38-98.compute-1.amazonaws.com	"}
PORT=${3:-8888}
ssh -i $PEM_FILE_PATH ubuntu@$EC2_DNS -L $PORT:localhost:$PORT
