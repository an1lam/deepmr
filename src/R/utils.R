fit_egger <- function(exp_eff_sizes, exp_std_errs, out_eff_sizes, out_std_errs) {
  MRInputObject <- mr_input(
    bx = exp_eff_sizes,
    bxse = exp_std_errs,
    by = out_eff_sizes,
    byse = out_std_errs
  )
  if (length(exp_eff_sizes) > 10) {
    EggerObject <- mr_egger(MRInputObject, aha=params$alpha, robust=TRUE, iterations = 10000)
  } else {
    EggerObject <- mr_egger(MRInputObject, alpha=params$alpha)
  }
  result <- c(
    EggerObject$Estimate,
    EggerObject$CILower.Est,
    EggerObject$CIUpper.Est,
    EggerObject$I.sq,
    EggerObject$Pleio.pval,
    EggerObject$StdError.Est,
    EggerObject$Intercept,
    EggerObject$Pvalue.Est
  )
  return(result)
}

fit_ivw <- function(exp_eff_sizes, exp_std_errs, out_eff_sizes, out_std_errs) {
  MRInputObject <- mr_input(
    bx = exp_eff_sizes,
    bxse = exp_std_errs,
    by = out_eff_sizes,
    byse = out_std_errs
  )  
  IvwObject <- mr_ivw(MRInputObject, alpha = params$alpha)
  result <- c(
    IvwObject$Estimate,
    IvwObject$CILower,
    IvwObject$CIUpper,
    IvwObject$StdError,
    IvwObject$Pvalue
  )
  return(result)
}

fit_mbe <- function(exp_eff_sizes, exp_std_errs, out_eff_sizes, out_std_errs) {
  MRInputObject <- mr_input(
    bx = exp_eff_sizes,
    bxse = exp_std_errs,
    by = out_eff_sizes,
    byse = out_std_errs
  )  
  MbeObject <- mr_mbe(MRInputObject, alpha = params$alpha)
  result <- c(
    MbeObject$Estimate,
    MbeObject$CILower,
    MbeObject$CIUpper,
    MbeObject$StdError,
    MbeObject$Pvalue
  )
  return(result)
}

fit_raps <- function(exp_eff_sizes, exp_std_errs, out_eff_sizes, out_std_errs) {
  RapsObject <- mr.raps(
    exp_eff_sizes, 
    out_eff_sizes, 
    exp_std_errs, 
    out_std_errs
  )
  result <- c(
    RapsObject$beta.hat,
    RapsObject$beta.se
  )
  return(result)
}


run_mr <- function(seq_veffs, n_seqs, mr_method) {
  ivw_vals <- list()
  egger_vals <- list()
  raps_vals <- list()
  
  if (mr_method == "egger") {
    n_columns <- 8
  } else if (mr_method == "ivw") {
    n_columns <- 5
  } else if (mr_method == "mbe") {
    n_columns <- 5
  } else if (mr_method == "raps") {
    n_columns <- 2
  } else {
    stop(str_c("Invalid MR method:", mr_method, sep=" "))
  }
  
  mr_results <- matrix(ncol = n_columns, nrow = n_seqs)
  
  for (seq in (1:n_seqs)) {
    veffs <- subset(seq_veffs, seq_num == seq)
    if (nrow(veffs) > 2) {
      bxt <- unlist(veffs["X_pred_mean"])
      bxset <- unlist(veffs["X_pred_var"])
      byt <- unlist(veffs["Y_pred_mean"])
      byset <- unlist(veffs["Y_pred_var"])
      
      if (mr_method == "egger") {
        mr_res <- tryCatch(
          fit_egger(bxt, bxset, byt, byset),
          error=function(err) {
            message(err)
            return(NA)
          }
        )
      } else if (mr_method == "ivw") {
        mr_res <- fit_ivw(bxt, bxset, byt, byset)
      } else if (mr_method == "mbe") {
        mr_res <- fit_mbe(bxt, bxset, byt, byset)
      } else if (mr_method == "raps") {
        mr_res <- fit_raps(bxt, bxset, byt, byset)
      }
      mr_results[seq,] <- mr_res
    }
  }
  
  if (mr_method == "egger") {
    colnames(mr_results) <- paste(
      "egger", 
      c(
        "ce",
        "cil",
        "ciu",
        "i.sq",
        "pleio",
        "std",
        "int",
        "pval"
      ),
      sep="."
    )
  } else if (mr_method == "ivw") {
    colnames(mr_results) <- paste(
      "ivw", 
      c(
        "ce",
        "cil",
        "ciu",
        "std",
        "pval"
      ),
      sep = "."
    )
  } else if (mr_method == "mbe") {
    colnames(mr_results) <- paste(
      "mbe", 
      c(
        "ce",
        "cil",
        "ciu",
        "std",
        "pval"
      ),
      sep = "."
    )
  } else if (mr_method == "raps") {
    colnames(mr_results) <- paste(
      "raps", 
      c(
        "ce",
        "std"
      ),
      sep = "."
    )
  }
  mr_results <- as_tibble(mr_results)
  return(mr_results)
}

compute_baseline_ces <- function(seq_veffs, n_seqs) {
  
  naive_ces <- c()
  
  for (seq in (1:n_seqs)) {
    veffs <- subset(seq_veffs, seq_num == seq)
    if (nrow(veffs) > 2) {
      bxt <- unlist(veffs["X_pred_mean"])
      byt <- unlist(veffs["Y_pred_mean"])
      naive_ces <- append(naive_ces, mean(byt / bxt))
    }
  }
  return(naive_ces)
}

measure_confidence_interval_calibration <- function(cis_true_vals) {
  true_val_captured <- cis_true_vals[which(
    (cis_true_vals$true_val >= cis_true_vals$lower) & 
      (cis_true_vals$true_val <= cis_true_vals$upper)
  ),]
  true_val_not_captured <- cis_true_vals[which(
    !((cis_true_vals$true_val >= cis_true_vals$lower) & 
        (cis_true_vals$true_val <= cis_true_vals$upper))
  ),]
  assertthat::are_equal(
    nrow(true_val_captured) + nrow(true_val_not_captured), nrow(cis_true_vals)
  )
  return(nrow(true_val_captured) / nrow(cis_true_vals))
}

ces_kde_plot <- function(ces, save_plots = F, xrange = NA) {
  p <- ggplot() +
    geom_density(aes(x = ces)) +
    xlab("Causal Effect") +
    ylab("Density")
  if (!is.na(xrange)) {
    p <- p + xlim(xrange)
  }
  return(p)
}

make_mr_plot <- function(eff_sizes) {
  bxt <- unlist(eff_sizes["X_pred_mean"])
  bxset <- unlist(eff_sizes["X_pred_var"])
  byt <- unlist(eff_sizes["Y_pred_mean"])
  byset <- unlist(eff_sizes["Y_pred_var"])
  MRInputObject <- mr_input(
    bx = bxt,
    bxse = bxset,
    by = byt,
    byse = byset,
  )
  return(mr_plot(MRInputObject, interactive=F))
}

dmode <- function(x) {
  den <- density(x, kernel=c("gaussian"))
  ( den$x[den$y==max(den$y)] )   
}