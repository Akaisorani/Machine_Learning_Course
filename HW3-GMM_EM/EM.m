%EM
N1=1000;
N2=1000;
mu1=[2,2];
mu2=[1,5];
sigma1=[2,0;0,0.1];
sigma2=[1,0.5;0.5,1];

X1=mvnrnd(mu1,sigma1,N1);
X2=mvnrnd(mu2,sigma2,N2);
X=[X1;X2];
% hold on;
% scatter(X1(:,1),X1(:,2),'.');
% scatter(X2(:,1),X2(:,2),'.');
[label,model,llh]=emm(X);
hold on;
showpoints(label,model,X);
showbounds(label,model,X);
hold off;
function [label,model,llh]=emm(X)
    tol=1e-6;
    maxiter=20;
    [N,K]=size(X);
    label=ceil(K*rand(N,1));
    R=zeros(N,K);
    for i=1:N
        R(i,label)=1;
    end
    model=maximization(X,R);
    for iter=2:maxiter
        [R,llh]=expectation(X,model);
        [~,label(1,:)]=max(R,[],2);
        R=R(:,unique(label));
        if iter~=2 && abs(lastllh-llh)<tol*abs(llh);break;end;
        lastllh=llh;
        model=maximization(X,R);
    end
end

function expectation(X,model)
    n=size(X,1);
    k=size(model.mu,2);
    R=zeros(n,k);
    for i=1:k
        R(:,i)=mvnpdf(X,model.mu(:,i),model.Sigma(:,:,i));
    end;
    R=bsxfun(@times,R,model.w);
    deno=sum(R,2);
    llh=sum(log(deno))/n;
    R=bsxfun(@rdivide,R,T);
end

function model=maxmization(X,R)
    [n,d]=size(X);
    k=size(R,2);
    nk=sum(R,1);
    w=nk/n;
    mu=bsxfun(@rdivide,X'*R,nk);
    Sigma=zeros(d,d,k);
    for i=1:k
        Xo=bsxfun(@minus,X,mu(:,i)');
        for j=1:n
            Sigma(:,:,i)=Sigma(:,:,i)+Xo(j,:)'*Xo(j,:)*R(j,k);
        end
        Sigma(:,:,i)=Sigma(:,:,i)./nk(i);
    end
    model.mu=mu;
    model.Sigma=Sigma;
    model.w=w;
end

function showbounds(label,model,X)
    [x,y]=meshgrid(linspace(min(X(:,1)),max(X(:,1)),500),linspace(min(2(:,1)),max(X(:,2)),500));
    k=size(model.mu,2);
    for i=1:k
        %points_k=X(find(label==i),:);
        %scatter(points_k(:,1),points_k(:,2));
        z=mvnpdf([x,y],model.mu(:,i)',model.Sigma(:,:,i));
        contour(x,y,z,'LevelList',[0.5]);
    end
end

function showpoints(label,model,X)
    [x,y]=meshgrid(linspace(min(X(:,1)),max(X(:,1)),500),linspace(min(2(:,1)),max(X(:,2)),500));
    k=size(model.mu,2);
    for i=1:k
        points_k=X((label==i),:);
        scatter(points_k(:,1),points_k(:,2));
    end
end
    
    
