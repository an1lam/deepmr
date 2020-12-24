Remember that installing `tidyverse`, `ks`, and `MendelianRandomization` will require `apt-get install`-ing a bunch of Linux packages like `libcurl`, `openssl`, and more.

Also, `mvtnorm` doesn't work with R <=3.4.X so you have to upgrade R. This requires doing the gpg solution to the keyserver download described [here](https://cloud.r-project.org/bin/linux/ubuntu/README.html) but modifed as described by [this Ubuntu stackexchange post](https://unix.stackexchange.com/a/399091). In practice this looks something like running the following series of commands:
