\name{PlotAdaptiveReg}
\alias{PlotAdaptiveReg}
\title{
R^2 / Number of latent variables plot
}
\description{
This function produces a plot of the R^2 estimated by cross-validation as a function of the number of latent variables included in the fitting. The vertical red line indicates the optimal R^2 and the corresponding number of latent variables.
}
\usage{
PlotAdaptiveReg(fit)
}
\arguments{
  \item{fit}{An object containing a fitted model obtained by the                     \code{FitAdaptiveReg} function.
}
}
\examples{
library(prospectr)
data(NIRsoil)
indNA = which(is.na(NIRsoil$Nt))
NIRNt = list(x=NIRsoil$spc[-indNA,],y=NIRsoil$Nt[-indNA])
fit = FitAdaptiveReg(NIRNt$x,NIRNt$y,nvmax=100)
PlotAdaptiveReg(fit)
}
