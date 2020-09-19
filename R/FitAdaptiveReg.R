FitAdaptiveReg = function(X,Y,nvmax=min(c(nrow(X)-2,ncol(X),100)),
                          nfolds=min(c(10,nrow(X)))){
  if((nfolds<3)||(nfolds>length(Y))){
    stop("Error, nfolds must be between 3 and n")
  }
  Y = matrix(Y)
  res.ZgMoments = ZgMoments(X,Y)
  folds = split(sample(0:(nrow(X)-1)),floor(((1:nrow(X))-1)/(nrow(X)/nfolds)))
  resCV = CVpredZg(X,Y,folds,nvmax)
  R2 = round(as.vector(cor(Y[unlist(folds)+1],resCV)^2),2)
  nv = which.max(R2)
  res = list(ZgMoments=res.ZgMoments,nv=nv,R2=R2,X=X,Y=Y)
  return(res)
}
