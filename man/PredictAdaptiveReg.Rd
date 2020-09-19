\name{PredictAdaptiveReg}
\alias{PredictAdaptiveReg}
\title{
Prediction of new values using the adaptive regression predictor
}
\description{
This function predicts the values of the response variable associated to new observations of the explanatory variables.
}
\usage{
PredictAdaptiveReg(fit,Xtest)
}
\arguments{
  \item{fit}{An object containing a fitted model obtained by the
  \code{FitAdaptiveReg} function
  }
  \item{Xtest}{A matrix with p columns containing the new observations of the
  explanatory variables.}
}
\value{
A vector containing the predicted values of the response variable.
}

\examples{
library(prospectr)
data(NIRsoil)
indNA = which(is.na(NIRsoil$Nt))
NIRNt = list(x=NIRsoil$spc[-indNA,],y=NIRsoil$Nt[-indNA])
ind.test = sample(1:length(NIRNt$y),50)
fit = FitAdaptiveReg(NIRNt$x[-ind.test,],NIRNt$y[-ind.test],nvmax=100)
pred = PredictAdaptiveReg(fit,NIRNt$x[ind.test,])
plot(NIRNt$y[ind.test],pred,pch=16,xlab="Y",ylab=expression(hat(Y)))
}
