N=10;	%训练样本数
M=10;	%测试样本数
m=3;	%阶数
alpha=1e-1;	%步长
lambda=0;	%惩罚项权重
maxdiff=1e-9;	%判定收敛

x=rand(N+M,1);	%x均匀分布
nx=(0:0.01:1)';
y_sin=sin(2*pi*x);	%原函数
y_gus=randn(size(x))*0.2;	%噪声
Y=y_sin+y_gus;	%生成数据

X=ones(size(x,1),m+1);
for i=1:m
    X(:,i+1)=x.^i;
end
Xtest=X(N+1:N+M,:);
Ytest=Y(N+1:N+M,:);
X=X(1:N,:);
Y=Y(1:N,:);
x=x(1:N);

lambda=0;
theta=My_regress(X,Y,alpha,lambda,maxdiff);	%普通梯度下降
lambda=1e-4;
theta2=My_regress(X,Y,alpha,lambda,maxdiff);	%加惩罚梯度下降
theta3=(X'*X)\X'*Y;								%解析解
theta4=conjgrad(X'*X,X'*Y,zeros(size(X,2),1));	%共轭梯度下降
nX=ones(size(nx,1),m+1);
for i=1:m
    nX(:,i+1)=nx.^i;
end
origin=plot(nx,sin(2*pi*nx),'g-');
hold on;
sample=plot(x,Y,'ro');
rec=plot(nx,nX*theta,'b-');
rec2=plot(nx,nX*theta2,'m-');
rec3=plot(nx,nX*theta3,'c-');
rec4=plot(nx,nX*theta4,'k-');
axis([0 1 -1.5 1.5]);
legend('原始图像','加噪采样','拟合结果','加惩罚项','解析解','共轭梯度');
hold off;

%测试样本
terr=errorTest(Xtest,theta,Ytest);
terr2=errorTest(Xtest,theta2,Ytest);
terr3=errorTest(Xtest,theta3,Ytest);
terr4=errorTest(Xtest,theta4,Ytest);
sprintf('error 梯度=%f 惩罚=%f 解析=%f 共轭=%f\n',terr,terr2,terr3,terr4)

function [res]=My_regress(X,Y,alpha,lambda,maxerr)

    theta=ones(size(X,2),1)*0;	%初始点

    cnt=0;
    err=-1;
    while (cnt<=1000000)	%迭代轮数上限
        cnt=cnt+1;
        td=(X'*(X*theta-Y))/size(X,1)+lambda*theta;	%梯度
        theta=theta-alpha*td;	%负梯度方向移动
        err_new=error(X,theta,Y,lambda);	%计算新误差
        if abs(err_new-err)<maxerr
            break;
        end
        if mod(cnt,10000)==0
            sprintf('round = %d, error = %f\n',cnt,err)	%输出中间结果
        end
        err=err_new;
    end
    
    res=theta;
    
    function err=error(X,theta,Y,lambda)
        result=X*theta;
        err=(result-Y)'*(result-Y)/2/size(X,1)+lambda*theta'*theta/2;
    end
end

function err=errorTest(X,theta,Y)	%测试数据
    result=X*theta;
    err=(result-Y)'*(result-Y)/2/size(X,1);
end

function [x]=conjgrad(A,b,x)	%共轭梯度下降
    cnt=0;
    r=b-A*x;
    p=r;
    r_old=r'*r;
    for i=1:length(b)*10
        Ap=A*p;
        alpha=r_old/(p'*Ap);
        x=x+alpha*p;	%向共轭梯度方向迭代
        r=r-alpha*Ap;
        r_new=r'*r;
        if sqrt(r_new)<1e-10
            break;
        end
        p=r+(r_new/r_old)*p;	%新的共轭搜索方向
        r_old=r_new;
        cnt=cnt+1;
        sprintf('round = %d, error = %.15f\n',cnt,r_new)
    end
end

