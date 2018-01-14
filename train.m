function [theta1,theta2,theta3]=train(X,y,noclass)   %X with ones
  a=(size(X,2))-1;
  b=400;  %HIDDEN LAYER 1 SIZE
  c=200; %HIDDEN LAYER 2 SIZE
  d=noclass;
  lambda=0.0; %LAMBDA VALUE
  epslon=0.001;  %EPSLON VALUE FOR INITIALIZATION
 

  initheta1=rand(b,a+1)*(2*epslon)-epslon;
  initheta2=rand(c,b+1)*(2*epslon)-epslon;
   initheta3=rand(d,c+1)*(2*epslon)-epslon;
  
  itu=[initheta1(:);initheta2(:);initheta3(:)];
  
  options=optimset('gradobj','on','maxiter',100);
  ftu=fmincg(@(t)(costcomp(X,y,t,lambda,a,b,c,d)),itu,options);
  
  theta1=reshape(ftu(1:(b*(a+1))),b,a+1);  %10x5
  ftu=ftu((b*(a+1))+1:end);
   theta2=reshape(ftu(1:(c*(b+1))),c,b+1); 
   ftu=ftu((c*(b+1))+1:end);
    theta3=reshape(ftu(1:(d*(c+1))),d,c+1); 
  
  
  
  
 
  end
  
  
  