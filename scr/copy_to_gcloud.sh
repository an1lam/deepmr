gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse ./src/*.py gregor-1-vm:~/project/src/
gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse ./src/pyx/*.py gregor-1-vm:~/project/src/pyx/
if [[ $1 == "-d" ]]; then
    gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse ./dat/ gregor-1-vm:~/project/dat/
fi
