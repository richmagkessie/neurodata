#' Train a NeuroData model on a data frame
#'
#' Encodes a tabular dataset as a neural network and saves it to a .kess file.
#' The file can be shared instead of the raw data.
#'
#' @param df A data.frame or tibble to encode.
#' @param path Output path for the .kess file (e.g. "health.kess").
#' @param epochs Number of training epochs. Default 500. Use 800+ for larger
#'   datasets or higher fidelity.
#' @param epsilon Differential privacy budget. Smaller = stronger privacy,
#'   lower fidelity. NULL = no DP (default).
#' @param delta DP failure probability. Default 1e-5.
#' @param password Password for AES-256-GCM encryption. NULL = no encryption
#'   (default). Set this to restrict who can load the model.
#' @param verbose Print training progress. Default TRUE.
#'
#' @return Invisibly returns \code{path}.
#'
#' @examples
#' \dontrun{
#' df <- read.csv("trial_data.csv")
#'
#' # No privacy — highest fidelity
#' neurodata_fit(df, "trial.kess", epochs = 800L)
#'
#' # With differential privacy
#' neurodata_fit(df, "trial_dp.kess", epochs = 1500L, epsilon = 5.0)
#'
#' # Encrypted (password-protected)
#' neurodata_fit(df, "trial_locked.kess", password = "secret123")
#' }
#' @export
neurodata_fit <- function(df, path,
                           epochs   = 500L,
                           epsilon  = NULL,
                           delta    = 1e-5,
                           password = NULL,
                           verbose  = TRUE) {
  nd$fit(
    df       = reticulate::r_to_py(as.data.frame(df)),
    path     = path,
    epochs   = as.integer(epochs),
    epsilon  = epsilon,
    delta    = delta,
    password = password,
    verbose  = verbose
  )
  invisible(path)
}


#' Load a NeuroData model and reconstruct the dataset
#'
#' Loads a .kess file and returns a data frame of reconstructed records.
#' The original CSV is never needed — only the model file.
#'
#' @param path Path to the .kess file.
#' @param password Decryption password if the model was saved with one.
#'   Default NULL.
#'
#' @return A \code{data.frame} of reconstructed records. Compatible with
#'   base R, dplyr, and tidyverse operations.
#'
#' @examples
#' \dontrun{
#' # Basic load
#' df <- neurodata_load("trial.kess")
#' head(df)
#' summary(df)
#'
#' # Load encrypted model
#' df <- neurodata_load("trial_locked.kess", password = "secret123")
#'
#' # Use with dplyr
#' library(dplyr)
#' df |> group_by(treatment_arm) |>
#'   summarise(success_rate = mean(trial_outcome == "Treatment Success"))
#'
#' df |> select(where(is.numeric)) |> cor()
#' }
#' @export
neurodata_load <- function(path, password = NULL) {
  frame <- nd$load(path = path, password = password)
  df    <- reticulate::py_to_r(frame$to_dataframe())
  as.data.frame(df)
}


#' Compute statistical fidelity of a reconstructed dataset
#'
#' Compares a reconstructed data frame against the original to quantify how
#' faithfully the model preserves statistical properties.
#' Numerical columns are evaluated with the Kolmogorov-Smirnov test;
#' categorical columns with distribution overlap (1 - total variation distance).
#'
#' @param original The original data.frame used to train the model.
#' @param reconstructed A data.frame returned by \code{neurodata_load()}.
#'
#' @return A named list with:
#' \describe{
#'   \item{overall_score}{Composite fidelity score 0–100 (higher = more faithful).}
#'   \item{columns}{Per-column scores and statistics.}
#' }
#'
#' @examples
#' \dontrun{
#' original      <- read.csv("trial_data.csv")
#' reconstructed <- neurodata_load("trial.kess")
#'
#' report <- neurodata_fidelity(original, reconstructed)
#' cat("Overall fidelity:", report$overall_score, "/ 100\n")
#' }
#' @export
neurodata_fidelity <- function(original, reconstructed) {
  report <- nd_fid$evaluate(
    original      = reticulate::r_to_py(as.data.frame(original)),
    reconstructed = reticulate::r_to_py(as.data.frame(reconstructed))
  )
  reticulate::py_to_r(report)
}


#' Print a formatted fidelity summary
#'
#' @param original The original data.frame.
#' @param reconstructed A data.frame from \code{neurodata_load()}.
#'
#' @return Invisibly returns the fidelity report list.
#' @export
nd_summary <- function(original, reconstructed) {
  report <- neurodata_fidelity(original, reconstructed)
  cat("\n--- NeuroData Fidelity Report ---\n")
  cat(sprintf("Overall Score:  %.1f / 100\n\n", report$overall_score))
  for (col in names(report$columns)) {
    info <- report$columns[[col]]
    cat(sprintf("  %-28s  score = %5.1f\n", col, info$col_score))
  }
  cat("---------------------------------\n")
  invisible(report)
}
