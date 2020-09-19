\name{FitAdaptiveReg}
\alias{FitAdaptiveReg}
\title{
Regression predictor based on an adaptive handling of dependence
}
\description{
This function constructs a predictor for regression problems using an adaptive handling of dependence. Dependence handling is controlled by the number of latent variables to be included in the construction of the prediction rule. The number of latent variables is selected using a cross-validation procedure maximizing the R^2.
}
\usage{
FitAdaptiveReg(X, Y, nvmax = min(c(nrow(X)-2,ncol(X),100)),
    nfolds = min(c(10,nrow(X))))
}
\arguments{
  \item{X}{
  A matrix of size n x p containing the observations of the explanatory variables
  }
  \item{Y}{
  A real-valued vector of length n containing the observations of the response
  }
  \item{nvmax}{
  An integer giving the maximum number of latent factors to take to construct
  the prediction rule. By default, the minimum value between \code{ncol(X)}, \code{nrow(X)-2} and 100 is taken. This number is reevaluated during the cross validation procedure and can thus be smaller than the value given at this point}
  \item{nfolds}{
  The number of folds to use to perform the cross-validation for the selection of the number of eigenvectors. Must be between 3 and n (for leave-one-out cross-validation). By default, leave-one-out cross validation is performed if n < 10, and 10-fold cross validation is performed if n >= 10.
  }
}
\value{
A list containing the following elements:
\item{ZgMoments}{A list containing several elements related to the moments of the latent variables involved in the calculation of the predictor}
\item{nv}{The optimal number of latent variables}
\item{R2}{The vector of values of the cross-validation R^2 (one value for each possible number of latent variables).}
}

\examples{
library(prospectr)
data(NIRsoil)
indNA = which(is.na(NIRsoil$Nt))
NIRNt = list(x=NIRsoil$spc[-indNA,],y=NIRsoil$Nt[-indNA])
fit = FitAdaptiveReg(NIRNt$x,NIRNt$y,nvmax=100) #100 latent factors at most
}
