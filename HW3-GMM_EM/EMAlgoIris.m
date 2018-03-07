%EM
data=textscan(fopen('bezdekIris.data'),'%f,%f,%f,%f,%s');
X=cell2mat(data(:,1:4));
Y=data(:,5);
Y=Y{1};
K=3;
[FT,Z,RE]=pcaForIris(X,3)
hold on;
[label,model,llh]=emm(X,K);
fprintf('class num %d\n',size(model.mu,2))
showpoints(label,model,Z);
%showbounds(label,model,Z);
hold off;
function [label,model,llh]=emm(X,initK)
    tol=1e-6;
    maxiter=100;
    N=size(X,1);
    K=initK;
    label=ceil(K*rand(N,1));
    R=zeros(N,K);
    for i=1:N
        R(i,label(i))=1;
    end
    model=maximization(X,R);
    for iter=2:maxiter
        [R,llh]=expectation(X,model);
        [~,label]=max(R,[],2);
        R=R(:,unique(label));
        fprintf('round=%d, llh=%d\n',iter,llh)
        if iter~=2 && abs(lastllh-llh)<tol*abs(llh);break;end;
        lastllh=llh;
        model=maximization(X,R);
    end
end

function [R,llh]=expectation(X,model)
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

function model=maximization(X,R)
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
        Sigma(:,:,i)=Sigma(:,:,i)./nk(i)+eye(d)*(1e-6);
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

function showpoints(label,model,X)
    k=size(model.mu,2);
    for i=1:k
        points_k=X((label==i),:);
        scatter3(points_k(:,1),points_k(:,2),points_k(:,3));
        %scatter(points_k(:,1),points_k(:,2));
    end
end

    
