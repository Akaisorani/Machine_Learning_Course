function [FT,Z,RE]=pcaForIris(X,K)
    mu=mean(X);
    Xcen=X-repmat(mu,size(X,1),1);
    C=cov(Xcen);
    [U,D]=eig(C);
    lambda=wrev(diag(D));
    U=fliplr(U);
    FT=U(:,1:K);
    Z=Xcen*FT;
    RE=Z*FT'+repmat(mu,size(X,1),1);
end