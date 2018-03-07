alpha=1e-3;
lambda=0;
rounds=1000;
epsilon=1e-6;

X=load('Breast_Cancer_Wisconsin_(Diagnostic)_Data_Set_UCI.txt');
Y=X(:,end);
X=[ones(size(X,1),1),X(:,1:end-1)];
N=size(X,1);

Xtest=X(round(N*0.7)+1:N,:);
Ytest=Y(round(N*0.7)+1:N,:);
Xtrain=X(1:round(N*0.7),:);
Ytrain=Y(1:round(N*0.7),:);

w=logi(Xtrain,Ytrain,lambda,rounds,epsilon);
wgrad=grad(Xtrain,Ytrain,alpha,lambda,rounds,epsilon);
lambda=1;
wlam=logi(Xtrain,Ytrain,lambda,rounds,epsilon);
wgradlam=grad(Xtrain,Ytrain,alpha,lambda,rounds,epsilon);

str=['loss-Newton = ',num2str(loss(Xtest,Ytest,w,0)),', fscore-Newton = ',num2str(fscore(Xtest,Ytest,w)),'\n','loss-Grad = ',num2str(loss(Xtest,Ytest,wgrad,0)),', fscore-Grad = ',num2str(fscore(Xtest,Ytest,wgrad)),'\n','loss-Newton-pun = ',num2str(loss(Xtest,Ytest,wlam,0)),', fscore-Newton-pun = ',num2str(fscore(Xtest,Ytest,wlam)),'\n','loss-Grad-pun = ',num2str(loss(Xtest,Ytest,wgradlam,0)),', fscore-Grad-pun = ',num2str(fscore(Xtest,Ytest,wgradlam)),'\n'];
fprintf(str);

disp('wnewton=');disp(w');
disp('wgrad=');disp(wgrad');
disp('wnewton-pun=');disp(wlam');
disp('wgrad-pun=');disp(wgradlam');

hold off;

function [zz]=getz(xx,yy,w)
    zz=w(1)+xx*w(2)+yy*w(3)+xx.*yy*w(4)+xx.*xx*w(5)+yy.*yy*w(6);
end

function [w]=logi(X,Y,lambda,rounds,epsilon)
   w=zeros(size(X,2),1);
   
   cs=0;
   while(cs<rounds)
       cs=cs+1;
       
       h=1./(1+exp(-X*w));
       g=X'*(h-Y)+lambda*w;
       A=diag(h.*(1-h));
       H=X'*A*X+lambda*eye(size(w,1));
       w=w-H\g;
       
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