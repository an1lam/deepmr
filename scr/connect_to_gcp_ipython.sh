ZONE=us-west1-b
PROJECT_ID=coms-4995-260215
INSTANCE_NAME=gregor-2-vm
PORT=${1:-8080}
echo "Connecting on port $PORT"
gcloud compute ssh --project $PROJECT_ID --zone $ZONE   $INSTANCE_NAME -- -L $PORT:localhost:$PORT
