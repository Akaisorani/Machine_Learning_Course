N=100;
K=2;
MU=[0,0,0];
SIGMA=[1,0,0;0,0.01,0;0,0,0.7];
X_origin=mvnrnd(MU,SIGMA,N);
hold on;
scatter3(X_origin(:,1),X_origin(:,2),X_origin(:,3));
axis([-5 5 -5 5 -5 5]);

X=X_origin;
mu=mean(X_origin);
X=X-repmat(mu,size(X_origin,1),1);
C=cov(X);
[U,D]=eig(C);
lambda=wrev(diag(D));
U=fliplr(U);
ft=U(:,1:K);
Z=X*ft;

X2=Z*ft'+repmat(mu,size(X_origin,1),1);
scatter3(X2(:,1),X2(:,2),X2(:,3),'.');
axis([-4 4 -4 4 -4 4]);
hold off;


