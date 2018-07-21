
%Author: Harish Mohan
%MNIST implementation using Neural networks BPN
load('image.mat');
ff=zeros(5000,10);

%Hot one Encoding
for i=1:5000
ff(i,y(i))=1;
end

input=size(X,1);
ip_shape=size(X,2);
hidden=25;
op_shape=10;
Theta1=rand(hidden,ip_shape)*2*1-1;
Theta2=rand(op_shape,hidden)*2*1-1;
% Theta1=rand(hidden,ip_shape)*2*1-1;
% Theta2=rand(1,hidden)*2*5;
b2=zeros(input,op_shape);
b1=zeros(input,hidden);
lambda=0.5;

for i = 1:1000
    %Feed Forward
    %Hidden layer compute
    z1=X*Theta1';
    z1=z1+b1;
    a1=1./(1+exp(-z1));
    %Output layer compute
    z2=a1*Theta2';
    z2=z2+b2;
    a2=1./(1+exp(-z2));
    
    if mod(i,100)==0
            xx=[];
            for j=1:5000
                xx=[xx;find(a2(j,:)==max(a2(j,:)))];
            end
           fprintf('Error Percentage : %d \n',mean(abs(xx - y)) * 100);
    end
    %BackProp
    %Error Layers
    dly=a2.*(1-a2).*(a2-ff);
    dlh=a1.*(1-a1).*(dly*Theta2);
    dlT2=-lambda/input*dly'*a1;
    dlb2=-lambda*dly;
    dlT1=-lambda/input*dlh'*X;
    dlb1=-lambda*dlh;
    %Updating values
    Theta1=Theta1+dlT1;
    Theta2=Theta2+dlT2;
    b1=b1+dlb1;
    b2=b2+dlb2;
end   
rp = randperm(input);

for i = 1:4
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    fprintf('\nNeural Network Prediction: %d (digit %d)\n', xx(rp(i)), mod(xx(rp(i)),10));
    
    % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end
