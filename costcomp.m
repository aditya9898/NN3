function [J,grad]=costcomp(X,y,thetaunrolled,lambda,a,b,c,d)  %4,10,10,3  without bias
  
  theta1=reshape(thetaunrolled(1:(b*(a+1))),b,a+1);  %10x5
  thetaunrolled=thetaunrolled((b*(a+1))+1:end);
   theta2=reshape(thetaunrolled(1:(c*(b+1))),c,b+1); %10x11
   thetaunrolled=thetaunrolled((c*(b+1))+1:end);
    theta3=reshape(thetaunrolled(1:(d*(c+1))),d,c+1);   %3x11
  
  
  
  
  a1=X';   %5x150
  z2=theta1*a1;  %10*150
  a2=[ones(1,length(X));sigmoid(z2)];  %11x150
   z3=theta2*a2;  %10*150
  a3=[ones(1,length(X));sigmoid(z3)];  %11x150
  z4=theta3*a3;  %3x150
  a4=sigmoid(z4);   %3x150
  J=0;
  m=length(X);
  for i=1:m
    yeff=zeros(d,1);
    yeff(y(i))=1;
    J=J+sum(((-yeff).*log(a4(:,i)))-((1-yeff).*log(1-a4(:,i))));
  end
    
    J=J/m;
    tt1=theta1(:,2:end);
    tt2=theta2(:,2:end);
    tt3=theta3(:,2:end);
    ttt=[tt1(:);tt2(:);tt3(:)];
    
    J=J+(lambda/m)*sum(ttt.^2);
    
    delta1=zeros(size(theta1));
    delta2=zeros(size(theta2));
    delta3=zeros(size(theta3));
    
    
    
    for j=1:m
      
       yeff=zeros(d,1);  %3x1
       yeff(y(j))=1;
      a1=X(j,:)';  %5x1
      z2=theta1*a1;  %10x1
      a2=[1;sigmoid(z2)];  
     z3=theta2*a2;  %10x1
      a3=[1;sigmoid(z3)];    %11x1
      z4=theta3*a3;  %3x1
      a4=sigmoid(z4);  %3x1
      
      
      sigma4=a4-yeff; %3x1
      sigma3=(theta3(:,2:end)'*sigma4).*a3(2:end).*(1-a3(2:end));  %10x1
       sigma2=(theta2(:,2:end)'*sigma3).*a2(2:end).*(1-a2(2:end));  %10x1
      
      
      
      delta1=delta1+sigma2*a1';  %10x5
      delta2=delta2+sigma3*a2'; 
     delta3=delta3+sigma4*a3';  %3x11
      end
      
      
      vert1=theta1;
      vert1(:,1)=0;
      vert2=theta2;
      vert2(:,1)=0;
      vert3=theta3;
      vert3(:,1)=0;
      D1=(1/m)*delta1+(lambda/m)*vert1;
      D2=(1/m)*delta2+(lambda/m)*vert2;
      D3=(1/m)*delta3+(lambda/m)*vert3;
    grad=[D1(:);D2(:);D3(:)];
    
    end
    
    
      
      
      
      
      
      
      
    
    
    
    
  
  