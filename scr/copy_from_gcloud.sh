if [[ $1 == "-s" ]]; then
    gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse gregor-2-vm:~/project/src/*.py src/
    gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse gregor-2-vm:~/project/src/pyx/*.py ./src/pyx/
fi
if [[ $1 == "-n" ]]; then
    gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse gregor-2-vm:~/project/src/*.ipynb src/
fi
if [[ $1 == "-d" ]]; then
    gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse gregor-2-vm:~/project/dat/ ./dat/ 
fi
if [[ $1 == "-r" ]]; then
    gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse gregor-2-vm:~/project/dat/means_and_uncertainties.csv ./dat/
fi
if [[ $1 == "-t" ]]; then
    gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse gregor-2-vm:~/project/**/*.tex ./doc/
fi
