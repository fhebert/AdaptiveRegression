\name{AdaptiveRegression-package}
\alias{AdaptiveRegression-package}
\docType{package}
\encoding{UTF-8}
\title{
  Regression rules based on an adaptive handling of dependence
}
\description{
This package provides functions to construct regression rules using an adaptive handling of dependence.
}
\author{
Florian Hébert, Mathieu Emily, David Causeur

Maintainer: Florian Hébert <florian.hebert@agrocampus-ouest.fr>
}
\keyword{Regression, Dependence, Prediction}
\examples{
  library(prospectr)
  data(NIRsoil)
  indNA = which(is.na(NIRsoil$Nt))
  NIRNt = list(x=NIRsoil$spc[-indNA,],y=NIRsoil$Nt[-indNA]) #remove NA
  NIRNt = list(x=NIRNt$x[-c(287,316,321,486,641,643),],
               y=NIRNt$y[-c(287,316,321,486,641,643)]) #remove outliers
  ind.test = sample(1:length(NIRNt$y),50)
  fit = FitAdaptiveReg(NIRNt$x[-ind.test,],NIRNt$y[-ind.test],nvmax=100)
  PlotAdaptiveReg(fit)
  pred = PredictAdaptiveReg(fit,NIRNt$x[ind.test,])
  plot(NIRNt$y[ind.test],pred,pch=16,xlab="Y",ylab=expression(hat(Y)))
}
