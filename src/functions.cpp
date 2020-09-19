//[[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include <stdint.h>
using namespace Rcpp;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXXd;
using Eigen::ArrayXd;
using Eigen::SelfAdjointEigenSolver;
using Rcpp::as;
using namespace std;

Eigen::MatrixXd SelectRows(Eigen::MatrixXd A, Eigen::VectorXd x){
    int n = x.size();
    int p = A.cols();
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(n,p);
    for(int i=0;i<n;i++){
        res.row(i) = A.row(x(i));
    }
    return(res);
}

Eigen::MatrixXd ListMatRowBind(List L){
    Eigen::MatrixXd X = L[0];
    int p = X.cols();
    int i = 1;
    int n = X.rows();
    for(i=1;i<L.size();i++){
        X = L[i];
        if(X.cols()<p){
            p = X.cols();
        }
        n = n+X.rows();
    }
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(n,p);
    int a = 0;
    for(i=0;i<L.size();i++){
        X = L[i];
        res.block(a,0,X.rows(),p) = X.block(0,0,X.rows(),p);
        a = a+X.rows();
    }
    return(res);
}

List SplitCV(Eigen::MatrixXd X, Eigen::MatrixXd Y, List CVFolds, int val){
    Eigen::MatrixXd Xval = SelectRows(X,CVFolds[val]);
    Eigen::MatrixXd Yval = SelectRows(Y,CVFolds[val]);
    List Xtrainlist;
    List Ytrainlist;
    int k = CVFolds.size();
    int j = 0;
    for(int i=0;i<k;i++){
        if(i!=val){
            Xtrainlist.insert(j,SelectRows(X,CVFolds[i]));
            Ytrainlist.insert(j,SelectRows(Y,CVFolds[i]));
            j++;
        }
    }
    Eigen::VectorXd indval = CVFolds[val];
    int nval = indval.size();
    int ntrain = X.rows()-nval;
    Eigen::MatrixXd Xtrain = ListMatRowBind(Xtrainlist);
    Eigen::MatrixXd Ytrain = ListMatRowBind(Ytrainlist);
    List XSplit, YSplit;
    XSplit["Xtrain"] = Xtrain;
    XSplit["Xval"] = Xval;
    YSplit["Ytrain"] = Ytrain;
    YSplit["Yval"] = Yval;
    List res;
    res["X"] = XSplit;
    res["Y"] = YSplit;
    return(res);
}

Eigen::MatrixXd Reshape(Eigen::VectorXd x){
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(x.size(),1);
    X = X.array()+x.array();
    return(X);
}

Eigen::MatrixXd CrossProd(Eigen::MatrixXd X){
    int n = X.cols();
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(n,n);
    res.selfadjointView<Eigen::Lower>().rankUpdate(X.transpose());
    res.triangularView<Eigen::StrictlyUpper>() = res.transpose();
    return (res);
}

Eigen::MatrixXd Var(Eigen::MatrixXd X){
    int n = X.rows();
    Eigen::MatrixXd xbar = X.colwise().mean();
    Eigen::MatrixXd u = Eigen::MatrixXd::Ones(n,1);
    X = X-u*xbar;
    Eigen::MatrixXd S = CrossProd(X);
    S = S/(n-1.0);
    return(S);
}

Eigen::MatrixXd Cov(Eigen::MatrixXd X, Eigen::MatrixXd Y){
    int n = X.rows();
    Eigen::MatrixXd xbar = X.colwise().mean();
    Eigen::MatrixXd ybar = Y.colwise().mean();
    Eigen::MatrixXd u = Eigen::MatrixXd::Ones(n,1);
    X = X-u*xbar;
    Y = Y-u*ybar;
    Eigen::MatrixXd S = X.transpose()*Y;
    S = S/(n-1.0);
    return(S);
}

Eigen::MatrixXd ColSd(Eigen::MatrixXd X){
    int n = X.rows();
    Eigen::MatrixXd xbar = X.colwise().mean();
    X = pow(X.array(),2);
    Eigen::MatrixXd x2bar = X.colwise().mean();
    Eigen::MatrixXd s = (x2bar.array()-pow(xbar.array(),2))*n/(n-1);
    s = pow(s.array(),0.5);
    return(s);
}

double Sd(Eigen::VectorXd x){
    int n = x.size();
    double xbar = x.mean();
    Eigen::VectorXd x2 = pow(x.array(),2);
    double x2bar = x2.mean();
    double s = pow((x2bar-pow(xbar,2))*n/(n-1),0.5);
    return(s);
}

List Eigs(Eigen::MatrixXd A, int rank){
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigdec;
    eigdec.compute(A);
    Eigen::MatrixXd U = eigdec.eigenvectors();
    Eigen::MatrixXd L = eigdec.eigenvalues();
    List res;
    res["values"] = L.colwise().reverse().block(0,0,rank,1);
    res["vectors"] = U.rowwise().reverse().block(0,0,A.rows(),rank);
    return(res);
}

//[[Rcpp::export]]
List ZgMoments(Eigen::MatrixXd X, Eigen::MatrixXd Y){
    Eigen::MatrixXd Sx = Var(X);
    Eigen::MatrixXd xbar = X.colwise().mean();
    Eigen::MatrixXd sxy = Cov(X,Y);
    double sy = Sd(Y);
    Eigen::MatrixXd S = Sx-(sxy*sxy.transpose())/pow(sy,2);
    Eigen::MatrixXd Ds = pow(S.diagonal().array(),0.5);
    Eigen::MatrixXd R = S.array()/(Ds*Ds.transpose()).array();
    int n = X.rows();
    int p = X.cols();
    int rank = std::min(n-2,p);
    Eigen::MatrixXd u = Eigen::MatrixXd::Ones(n,1);
    Eigen::MatrixXd Xc = (X-u*xbar).array()/(u*Ds.transpose()).array();
    List ev = Eigs(R,rank);
    Eigen::MatrixXd U = ev["vectors"];
    Eigen::MatrixXd L = ev["values"];
    Eigen::MatrixXd Z = Xc*U;
    Eigen::MatrixXd sxyDs = sxy.array()/Ds.array();
    Eigen::MatrixXd gamma = (U.transpose()*sxyDs).transpose();
    Eigen::MatrixXd Zg = Z.array()*(u*gamma).array();
    Eigen::MatrixXd gamma2 = pow(gamma.array(),2);
    gamma2.transposeInPlace();
    Eigen::MatrixXd VZg1 = L.array()*gamma2.array();
    Eigen::MatrixXd VarZg1 = Eigen::MatrixXd::Zero(VZg1.rows(),VZg1.rows());
    VarZg1.diagonal() = VZg1;
    Eigen::MatrixXd VarZg2 = (1/pow(sy,2))*(gamma2*gamma2.transpose());
    Eigen::MatrixXd VarZg = VarZg1+VarZg2;
    Eigen::MatrixXd CovYZg = gamma2;
    List res;
    res["sy"] = sy;
    res["Zg"] = Zg;
    res["xbar"] = xbar;
    res["Xc"] = Xc;
    res["S"] = S;
    res["sxy"] = sxy;
    res["U"] = U;
    res["L"] = L;
    res["Ds"] = Ds;
    res["Z"] = Z;
    res["gamma"] = gamma;
    res["VarZg"] = VarZg;
    res["CovYZg"] = CovYZg;
    return(res);
}

//[[Rcpp::export]]
Eigen::MatrixXd predZg(List res_ZgMoments, Eigen::MatrixXd Xtest, int nvmax){
    Eigen::MatrixXd xbar = res_ZgMoments["xbar"];
    Eigen::MatrixXd U = res_ZgMoments["U"];
    Eigen::MatrixXd L = res_ZgMoments["L"];
    Eigen::MatrixXd gamma = res_ZgMoments["gamma"];
    Eigen::MatrixXd Ds = res_ZgMoments["Ds"];
    int ntest = Xtest.rows();
    Eigen::MatrixXd u = Eigen::MatrixXd::Ones(ntest,1);
    Eigen::MatrixXd testXc = (Xtest-u*xbar).array()/(u*Ds.transpose()).array();
    Eigen::MatrixXd Z = testXc*U;
    Eigen::MatrixXd Zg = Z.array()*(u*gamma).array();
    Eigen::MatrixXd VarZg = res_ZgMoments["VarZg"];
    Eigen::MatrixXd CovYZg = res_ZgMoments["CovYZg"];
    int p = VarZg.cols();
    nvmax = std::min(nvmax,p);
    List ev = Eigs(VarZg,nvmax);
    Eigen::MatrixXd VarZg_L = ev["values"];
    Eigen::MatrixXd VarZg_U = ev["vectors"];
    Eigen::MatrixXd v = Eigen::MatrixXd::Zero(p,1);
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(p,p);
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(p,1);
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(testXc.rows(),nvmax);
    for(int i=0;i<nvmax;i++){
        v = VarZg_U.col(i);
        A = A+(1/VarZg_L(i))*(v*v.transpose());
        h = A*CovYZg;
        res.col(i) = Zg*h;
    }
    return(res);
}

//[[Rcpp::export]]
Eigen::MatrixXd predZgFinal(List res_ZgMoments, Eigen::MatrixXd Xtest, int nv){
    Eigen::MatrixXd xbar = res_ZgMoments["xbar"];
    Eigen::MatrixXd U = res_ZgMoments["U"];
    Eigen::MatrixXd L = res_ZgMoments["L"];
    Eigen::MatrixXd gamma = res_ZgMoments["gamma"];
    Eigen::MatrixXd Ds = res_ZgMoments["Ds"];
    int ntest = Xtest.rows();
    Eigen::MatrixXd u = Eigen::MatrixXd::Ones(ntest,1);
    Eigen::MatrixXd testXc = (Xtest-u*xbar).array()/(u*Ds.transpose()).array();
    Eigen::MatrixXd Z = testXc*U;
    Eigen::MatrixXd Zg = Z.array()*(u*gamma).array();
    Eigen::MatrixXd VarZg = res_ZgMoments["VarZg"];
    Eigen::MatrixXd CovYZg = res_ZgMoments["CovYZg"];
    List ev = Eigs(VarZg,nv);
    Eigen::MatrixXd VarZg_L = ev["values"];
    Eigen::MatrixXd VarZg_U = ev["vectors"];
    Eigen::MatrixXd A = VarZg_U*VarZg_L.cwiseInverse().asDiagonal()*VarZg_U.transpose();
    Eigen::MatrixXd h = A*CovYZg;
    Eigen::MatrixXd res = Zg*h;
    return(res);
}


//[[Rcpp::export]]
Eigen::MatrixXd CVpredZg(Eigen::MatrixXd X, Eigen::MatrixXd Y,
              List CVFolds, int nvmax){
    int n = X.rows();
    int p = X.cols();
    int nmin = n;
    int ntrain = n;
    for(int i=0;i<CVFolds.size();i++){
        Eigen::VectorXd foldi = CVFolds[i];
        int ntrain = n-foldi.size()-2;
        if(ntrain<nmin){
            nmin = ntrain;
        }
    }
    nvmax = std::min(nmin,nvmax);
    List res;
    for(int i=0;i<CVFolds.size();i++){
        List cvsplit = SplitCV(X,Y,CVFolds,i);
        List Xcvsplit = cvsplit["X"];
        List Ycvsplit = cvsplit["Y"];
        Eigen::MatrixXd Xtrain = Xcvsplit["Xtrain"];
        Eigen::MatrixXd Xval = Xcvsplit["Xval"];
        Eigen::MatrixXd Ytrain = Ycvsplit["Ytrain"];
        Eigen::MatrixXd Yval = Ycvsplit["Yval"];
        List res_ZgMoments = ZgMoments(Xtrain,Ytrain);
        Eigen::MatrixXd Yprednv = predZg(res_ZgMoments,Xval,nvmax);
        res.insert(i,Yprednv);
    }
    Eigen::MatrixXd matres = ListMatRowBind(res);
    return(matres);
}
