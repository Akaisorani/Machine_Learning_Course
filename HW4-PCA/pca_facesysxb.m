im1=imread('dsjxxx.png');
im1=rgb2gray(im1);
im2=imread('ysxbxxx.png');
im2=rgb2gray(im2);
X=[reshape(im1,1,size(im1,1)*size(im1,2));reshape(im2,1,size(im2,1)*size(im2,2))];
size(X)
N=size(X,1);
K=1;
figure;
subplot(1,2,1);
showimage(X,1,2);
title('½µÎ¬Ç°');
[FT,Z,RE]=pca(X,K);
subplot(1,2,2);
showimage(RE,1,2);
title({['½µÎª',num2str(K),'Î¬ºó']});

function [FT,Z,RE]=pca(X,K)
    mu=mean(X);
    
    %Xcen=double(X)-repmat(mu,size(X,1),1);
    Xcen=double(X);
    C=cov(Xcen);
    [U,D]=eig(C);
    lambda=wrev(diag(D));
    U=fliplr(U);
    FT=U(:,1:K);
    Z=Xcen*FT;
    %RE=Z*FT'+repmat(mu,size(X,1),1);
    RE=Z*FT';
end

function showimage(X,n,m)
    h=sqrt(size(X,2));
    w=h;
    immat=reshape(X(1,:),h,w);
    for i=2:n*m
        immat=cat(4,immat,reshape(X(i,:),h,w));
    end
    montage(immat,[-128 128],'size',[n m]);
end

