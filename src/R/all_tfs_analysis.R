---
  title: "Mendelian Randomization analysis of DeepSEA generated data"
output: html_notebook
---
```{r}
library(MendelianRandomization)
library(dplyr)
library(tidyverse)
```

```{r}
files <- list.files(path="../dat/res/", pattern="*.txt")

lapply(files, function(f) {
  print(f)
})
```

```{r}
# seq_predictions = read.csv(file.path(data_dir, "effect_sizes__20200430__comparison_new.csv"))
seq_predictions = read.csv(file.path(results_dir, "HepG2_HDAC2_mutagenesis_results.csv"))
is.nan.data.frame <- function(x)
do.call(cbind, lapply(x, is.nan))

seq_predictions[is.nan(seq_predictions)] <- 0
seq_predictions
```

```{r}
seq_predictions <- seq_predictions %>%
  mutate(
    is.negative = X_pred_mean < 0,
    X_pred_mean = ifelse(is.negative, -X_pred_mean, X_pred_mean),
    Y_pred_mean = ifelse(is.negative, -Y_pred_mean, Y_pred_mean),
    X_pred_var = X_pred_var,
    Y_pred_var = Y_pred_var
  ) %>%
  filter(
    X_pred_var > 0 & Y_pred_var > 0
  )
```


```{r}
ivw_vals = list()
egger_vals = list()

egger_result <- matrix(ncol=6, nrow=25)
ivw_result <- matrix(ncol=4, nrow=25)


for (seq in (1:25)) {
  seq_i_predictions = subset(seq_predictions, seq_num == seq)
  
  bxt <- unlist(seq_i_predictions["X_pred_mean"])
  bxset <- unlist(seq_i_predictions["X_pred_var"])
  byt <- unlist(seq_i_predictions["Y_pred_mean"])
  byset <- unlist(seq_i_predictions["Y_pred_var"])
  MRInputObject <- mr_input(bx = bxt,
                            bxse = bxset,
                            by = byt,
                            byse = byset)
  EggerObject <- mr_egger(MRInputObject)
  IVWObject <- mr_ivw(MRInputObject)
  # LMObject <- lm.fit(matrix(bxt), byt)
  # 
  egger_result[seq, ] <- c(EggerObject$Estimate, EggerObject$CILower.Est, EggerObject$CIUpper.Est, EggerObject$I.sq, EggerObject$Pleio.pval, EggerObject$StdError.Est)
  ivw_result[seq, ] <- c(IVWObject$Estimate, IVWObject$CILower, IVWObject$CIUpper, IVWObject$StdError)
}

colnames(egger_result) <- c("ce", "cil", "ciu", "i.sq", "pleio", "std")
egger_result <- as_tibble(egger_result)
colnames(ivw_result) <- c("ce", "cil", "ciu", "std")
ivw_result <- as_tibble(ivw_result)
```

```{r}
egger_result$i.sq
```

Now we have the causal effect estimates, confidence intervals, and (for Egger) pleiotropy p-values.
```{r}
ivw_result
egger_result
```

Let's summarize the IVW & Egger causal effect estimates as histograms. This will give us an idea of how heterogeneous the alleged causal relationships are at different regions of the genome.
```{r}
qplot(ce, data=ivw_result, binwidth=.25)
qplot(ce, data=egger_result, binwidth=.25)
```

Let's also look at the distribution of pleiotropy p-values for different sequences. If they're concentrated around .5, this means that the algorithm believes there to be very little pleiotropy.

```{r}
qplot(pleio, data=egger_result, binwidth=.04)
```

```{r}
p <- ggplot2::ggplot(ivw_result, aes(x=ce, y=std)) + geom_point()
p
```

```{r}
plot(bxt, bxset)
```