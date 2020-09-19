PlotAdaptiveReg = function(fit){
  par(mar=c(5.1,4.5,4.1,2.1))
  plot(fit$R2,type="l",xlab="Number of latent factors",ylab=expression(R^2),
       main=paste("Optimal number of latent factors: ",fit$nv,sep=""))
  abline(v=fit$nv,col="red",lty=2)
  par(mar=c(5.1, 4.1, 4.1, 2.1))
}
