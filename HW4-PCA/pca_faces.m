X=load('ex7faces.mat');
X=X.X;
N=size(X,1);
K=5;    %��ά��ά��

subplot(1,3,1);
showimage(X,5,5);
title('��άǰ');
[FT,Z,RE]=pca(X,K); %PCA
meanpsnr_snr=psnrs(RE,X,255);
subplot(1,3,2);
showimage(RE,5,5);
title({['��Ϊ',num2str(K),'ά��'],['PSNR=',num2str(meanpsnr_snr(1)),',SNR=',num2str(meanpsnr_snr(2))]});
subplot(1,3,3);
showimage(FT'*1024,5,5);
title('��������');
function [FT,Z,RE]=pca(X,K)
    mu=mean(X);
    Xcen=X-repmat(mu,size(X,1),1);  %���Ļ�
    C=cov(Xcen);    %Э�������
    [U,D]=eig(C);   %������ֵ����������
    lambda=wrev(diag(D));
    U=fliplr(U);
    FT=U(:,1:K);    %ȡ����ֵ����ǰk����������
    Z=Xcen*FT;      %��ά
    RE=Z*FT'+repmat(mu,size(X,1),1);    %��ԭ
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
