PredictAdaptiveReg = function(fit,Xtest){
  if(!is.matrix(Xtest)) stop("Xtest must be a matrix")
  if(ncol(Xtest)!=ncol(fit$X)) stop("Xtest must have the same number of columns as X")
  tmp = as.vector(predZgFinal(fit$ZgMoments,fit$X,fit$nv))
  xbar = mean(tmp)
  ybar = mean(fit$Y)
  covxy = cov(tmp,fit$Y)
  varx = var(tmp)
  b = covxy/varx
  a = ybar-b*xbar
  res = a[1]+b[1]*as.vector(predZgFinal(fit$ZgMoments,Xtest,fit$nv))
  return(res)
}
