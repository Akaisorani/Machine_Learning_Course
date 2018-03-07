%EM
N1=100; %生成数据参数
N2=100;
N3=100;
K=3;
mu1=[2,2];
mu2=[1,5];
mu3=[4,4];
sigma1=[2,0;0,0.1];
sigma2=[1,0.5;0.5,1];
sigma3=[3,-0.5;-0.5,0.3];

X1=mvnrnd(mu1,sigma1,N1);
X2=mvnrnd(mu2,sigma2,N2);
X3=mvnrnd(mu3,sigma3,N3);
X=[X1;X2;X3];
% hold on;
% scatter(X1(:,1),X1(:,2),'.');
% scatter(X2(:,1),X2(:,2),'.');
hold on;
[label,model,llh]=emm(X,K); %EM算法
fprintf('class num %d\n',size(model.mu,2))
showpoints(label,model,X);
showbounds(label,model,X);
hold off;
function [label,model,llh]=emm(X,initK)
    tol=1e-6;   %收敛标准，相对误差
    maxiter=100;    %最大迭代次数
    N=size(X,1);
    K=initK;    %初始分类数
    label=ceil(K*rand(N,1));    %初始化
    R=zeros(N,K);
    for i=1:N
        R(i,label(i))=1;
    end
    model=maximization(X,R);
    for iter=2:maxiter
        [R,llh]=expectation(X,model);   %E-step
        [~,label]=max(R,[],2);  %整理分类标签
        R=R(:,unique(label));   %删除空分类
        fprintf('round=%d, llh=%d\n',iter,llh)
        if iter~=2 && abs(lastllh-llh)<tol*abs(llh);break;end;
        lastllh=llh;
        model=maximization(X,R);    %M-step
    end
end

function [R,llh]=expectation(X,model)   %E-step
    n=size(X,1);
    k=size(model.mu,2);
    R=zeros(n,k);
    for i=1:k
        R(:,i)=mvnpdf(X,model.mu(:,i)',model.Sigma(:,:,i));
    end;
    R=bsxfun(@times,R,model.w);
    deno=sum(R,2);
    llh=sum(log(deno))/n;
    R=bsxfun(@rdivide,R,deno);
end

function model=maximization(X,R)    %M-step
    [n,d]=size(X);
    k=size(R,2);
    nk=sum(R,1);
    w=nk/n;
    mu=bsxfun(@rdivide,X'*R,nk);
    Sigma=zeros(d,d,k);
    for i=1:k
        Xo=bsxfun(@minus,X,mu(:,i)');
        for j=1:n
            Sigma(:,:,i)=Sigma(:,:,i)+Xo(j,:)'*Xo(j,:)*R(j,i);
        end
        Sigma(:,:,i)=Sigma(:,:,i)./nk(i);
    end
    model.mu=mu;
    model.Sigma=Sigma;
    model.w=w;
end

function showbounds(label,model,X)
    [x,y]=meshgrid(linspace(min(X(:,1)),max(X(:,1)),500),linspace(min(X(:,2)),max(X(:,2)),500));
    k=size(model.mu,2);
    for i=1:k
        z=mvnpdf([x(:),y(:)],model.mu(:,i)',model.Sigma(:,:,i));
        z=reshape(z,size(x,1),size(x,2));
        contour(x,y,z,'Levellist',[max(max(z))/4]);
    end
end

function showbounds2(label,model,X)
    [x,y]=meshgrid(linspace(min(X(:,1)),max(X(:,1)),500),linspace(min(X(:,2)),max(X(:,2)),500));
    k=size(model.mu,2);
    z=zeros(size(x,1)*size(x,2),1);
    for i=1:k
        z=z+mvnpdf([x(:),y(:)],model.mu(:,i)',model.Sigma(:,:,i));
    end
    z=reshape(z,size(x,1),size(x,2));
    contour(x,y,z);
end


function showpoints(label,model,X)
    k=size(model.mu,2);
    for i=1:k
        points_k=X((label==i),:);
        scatter(points_k(:,1),points_k(:,2));
    end
end
    
    
