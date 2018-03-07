mu1=[2,2];  %0
mu2=[1,5];  %1
sigma1=[2,0;0,2];
sigma2=[1,0;0,1];
N=2000;
N1=round(N*0.7);	%70%反例
N2=N-N1;

alpha=1e-3;		%学习率
lambda=0;		%惩罚项权重
rounds=1000;	%最大轮数
epsilon=1e-6;	%判定收敛误差

X1=mvnrnd(mu1,sigma1,N1);
X1=[ones(size(X1,1),1),X1,X1(:,1).*X1(:,2),X1(:,1).*X1(:,1),X1(:,2).*X1(:,2)];
X2=mvnrnd(mu2,sigma2,N2);
X2=[ones(size(X2,1),1),X2,X2(:,1).*X2(:,2),X2(:,1).*X2(:,1),X2(:,2).*X2(:,2)];
X=[X1;X2];
Y=[zeros(size(X1,1),1);ones(size(X2,1),1)];
r=randperm(size(X,1));
X=X(r,:);
Y=Y(r,:);

%数据划分为测试集，训练集
Xtest=X(round(N*0.7)+1:N,:);
Ytest=Y(round(N*0.7)+1:N,:);
Xtrain=X(1:round(N*0.7),:);
Ytrain=Y(1:round(N*0.7),:);

w=logi(Xtrain,Ytrain,lambda,rounds,epsilon);
wgrad=grad(Xtrain,Ytrain,alpha,lambda,rounds,epsilon);
lambda=1;
wlam=logi(Xtrain,Ytrain,lambda,rounds,epsilon);
wgradlam=grad(Xtrain,Ytrain,alpha,lambda,rounds,epsilon);

hold on;
scatter(X1(:,2),X1(:,3),'x');
scatter(X2(:,2),X2(:,3),'filled');

info={['loss-Newton = ',num2str(loss(Xtest,Ytest,w,0)),', fscore-Newton = ',num2str(fscore(Xtest,Ytest,w))];
    ['loss-Grad = ',num2str(loss(Xtest,Ytest,wgrad,0)),', fscore-Grad = ',num2str(fscore(Xtest,Ytest,wgrad))];
    ['loss-Newton-pun = ',num2str(loss(Xtest,Ytest,wlam,0)),', fscore-Newton-pun = ',num2str(fscore(Xtest,Ytest,wlam))];
    ['loss-Grad-pun = ',num2str(loss(Xtest,Ytest,wgradlam,0)),', fscore-Grad-pun = ',num2str(fscore(Xtest,Ytest,wgradlam))];};
title(info);

[xx,yy]=meshgrid(linspace(min(X(:,2)),max(X(:,2)),500),linspace(min(X(:,3)),max(X(:,3)),500));
zz1=getz(xx,yy,w);
zz2=getz(xx,yy,wgrad);
zz3=getz(xx,yy,wlam);
zz4=getz(xx,yy,wgradlam);
[C1,h1]=contour(xx,yy,zz1,'LevelList',[0],'color','m');
[C2,h2]=contour(xx,yy,zz2,'LevelList',[0],'color','k');
% [C3,h3]=contour(xx,yy,zz3,'LevelList',[0],'color','c');
% [C4,h4]=contour(xx,yy,zz4,'LevelList',[0],'color','g');
legend('0','1','牛顿法','梯度下降');
%legend('0','1','牛顿法','梯度下降','牛顿法（惩罚）','梯度下降（惩罚）');
disp('wnewton=');disp(w');
disp('wgrad=');disp(wgrad');
disp('wnewton-pun=');disp(wlam');
disp('wgrad-pun=');disp(wgradlam');

hold off;

function [zz]=getz(xx,yy,w)
    zz=w(1)+xx*w(2)+yy*w(3)+xx.*yy*w(4)+xx.*xx*w(5)+yy.*yy*w(6);
end

function [w]=logi(X,Y,lambda,rounds,epsilon)	%logistic回归
   w=zeros(size(X,2),1);
   
   cs=0;
   while(cs<rounds)
       cs=cs+1;
       
       h=1./(1+exp(-X*w));
       g=X'*(h-Y)+lambda*w;		%梯度
       A=diag(h.*(1-h));
       H=X'*A*X+lambda*eye(size(w,1));	%海森矩阵
       w=w-H\g;			%迭代
       
       fprintf('round %d, loss %f\n',cs,loss(X,Y,w,lambda))
       if(sqrt(g'*g)<epsilon)
           break;
       end       
   end
end

function [w]=grad(X,Y,alpha,lambda,rounds,epsilon)
   w=zeros(size(X,2),1);
   
   cs=0;
   while(cs<rounds)
       cs=cs+1;
       
       h=1./(1+exp(-X*w));
       g=X'*(h-Y)+lambda*w;
       w=w-alpha*g;
       
       fprintf('round %d, loss %f\n',cs,loss(X,Y,w,lambda))
       if(sqrt(g'*g)<epsilon)
           break;
       end       
   end
end

function [ret]=loss(X,Y,w,lambda)
    h=1./(1+exp(-X*w));
    ret=-Y'*log(h)-(1-Y')*log(1-h)+lambda*(w'*w)/2;
end

function [score]=fscore(X,Y,w)
    h=1./(1+exp(-X*w));
    h(h>=0.5)=1;
    h(h<0.5)=0;
    precision=length(find(h.*Y))/length(find(h));
    recall=length(find(h.*Y))/length(find(Y));
    score=2*precision*recall/(precision+recall);
end