EC2_DNS=${1:-"ec2-54-157-160-62.compute-1.amazonaws.com	"}
PEM_FILE_PATH=${2:-"~/StephenDAK_Dell.pem"}
ssh -i $PEM_FILE_PATH ubuntu@$EC2_DNS
