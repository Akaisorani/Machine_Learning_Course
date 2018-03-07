X=load('ex7faces.mat');
X=X.X;
N=size(X,1);
K=5;    %降维后维数

subplot(1,3,1);
showimage(X,5,5);
title('降维前');
[FT,Z,RE]=pca(X,K); %PCA
meanpsnr_snr=psnrs(RE,X,255);
subplot(1,3,2);
showimage(RE,5,5);
title({['降为',num2str(K),'维后'],['PSNR=',num2str(meanpsnr_snr(1)),',SNR=',num2str(meanpsnr_snr(2))]});
subplot(1,3,3);
showimage(FT'*1024,5,5);
title('特征向量');
function [FT,Z,RE]=pca(X,K)
    mu=mean(X);
    Xcen=X-repmat(mu,size(X,1),1);  %中心化
    C=cov(Xcen);    %协方差矩阵
    [U,D]=eig(C);   %求特征值和特征向量
    lambda=wrev(diag(D));
    U=fliplr(U);
    FT=U(:,1:K);    %取特征值最大的前k个特征向量
    Z=Xcen*FT;      %降维
    RE=Z*FT'+repmat(mu,size(X,1),1);    %还原
end

function showimage(X,n,m)
    h=sqrt(size(X,2));
    w=h;
    immat=reshape(X(1,:),h,w);
    ed=min(size(X,1),n*m);
    for i=2:ed
        immat=cat(4,immat,reshape(X(i,:),h,w));
    end
    montage(immat,[-128 128],'size',[n m]);
end

function [PSNR_SNR]=psnrs(X1,X2,peakval)
    N=size(X1,1);
    psnrsum=[0,0];
    for i=1:N
        [psnri,snri]=psnr(X1(i,:),X2(i,:),peakval);
        psnrsum=psnrsum+[psnri,snri];
    end
    PSNR_SNR=psnrsum/N;
end
    

% function [PSNR]=psnr(f1,f2,B)
%     MAX=2^B-1;
%     MSE=sum(sum((f1-f2).^2))/(h*w);
%     PSNR=20*log10(MAX/sqrt(MSE));
% end
