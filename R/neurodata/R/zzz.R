nd      <- NULL
nd_fid  <- NULL

.onLoad <- function(libname, pkgname) {
  nd     <<- reticulate::import("neurodata",         delay_load = TRUE)
  nd_fid <<- reticulate::import("neurodata.fidelity", delay_load = TRUE)
}

.onAttach <- function(libname, pkgname) {
  if (!reticulate::py_module_available("neurodata")) {
    packageStartupMessage(
      "Python package 'neurodata' not found.\n",
      "Install it with:\n",
      "  reticulate::py_install(c('torch','pandas','numpy','scipy',\n",
      "                            'scikit-learn','cryptography'), pip = TRUE)\n",
      "Then clone https://github.com/richmagkessie/neurodata and run:\n",
      "  reticulate::py_install('.', pip = TRUE)"
    )
  }
}
