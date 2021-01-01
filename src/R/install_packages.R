# Note: Installing these packages on a new machine typically requires
# setting up OpenSSL:
# 
#   On Linux (Debian): sudo apt install r-cran-openssl
#  
# and libxml:
#
#   On Linux: sudo apt install libxml2-dev
#
# Pre-requisites
install.packages(c("readr", "iterpc", "plotly", "mvtnorm", "codetools"))

# R must-haves
install.packages("tidyverse")
install.packages("dplyr")

# 3rd party packages
install.packages("doSNOW")
install.packages("MendelianRandomization")
install.packages("mr.raps")
install.packages("ks")
install.packages("ggplot2")
install.packages("stringr")
install.packages("foreach")
install.packages("meta")
