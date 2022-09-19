function [x,fval,t,FunVals] = own_dsadam(cfun,x0,X,FoldLabels,StopTol,MaxFunEvals)

% Own implementation of Adam optimizer. (Kingma & Ba Algorithm 1).
% Take minibatch based steps by sampling input data X using Distribution
% optimal sampling.
% Author: Tommi Karkkainen (April, 2020)

%
% Stepsize, moment update parameters, and epsilon (square root of machine eps).
%
alpha = 0.001;
beta1 = 0.9;
beta2 = 0.999;
epsilon = sqrt(eps);

%
% In the autoencoding tests we encounter many times situations where one 
% do not need to improve the initial solution. Testing such a case below
% and acting accordingly.
%
fval0 = cfun(x0,X);
if fval0 < epsilon
    x = x0;
    fval = fval0;
    t = 0;
    FunVals = [];
    return
end
StopTol = fval0*StopTol;

x = x0;
FunVals = fval0;
nfolds = max(FoldLabels);

% Initialize moment vectors
n = length(x0);
m = zeros(n,1);
v = zeros(n,1);

t = 0;
StopCrit = 0;

while ~StopCrit
    t = t + 1;
    
    XF = X(FoldLabels == randi(nfolds),:);
    [fval,grad] = cfun(x,XF);

    FImp = FunVals(end) - fval;
    FunVals = [FunVals; fval];
    
    %
    m = beta1*m + (1 - beta1)*grad;
    v = beta2*v + (1 - beta2)*grad.^2;
    %
    mhat = m/(1 - beta1^t);
    vhat = v/(1 - beta2^t);
    update = alpha*mhat./(sqrt(vhat) + epsilon);
    %
    x = x - update;
    %
    StopCrit = ((FImp > 0) && (FImp < StopTol)) || (t >= MaxFunEvals);
end

fval = cfun(x,X);
FunVals = [FunVals; fval];

end
