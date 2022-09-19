function [W1,W2,ae_err] = Train1HidFFN(X,beta,W10,W20,flag)
% Solves layer matrices W1 and W2 for one hidden layer FFN based autoencoder
% using regularized LSE formulation and limited memory bfgs solver.
% Author: T. Karkkainen, April 2020
%
[N,n0] = size(X);
[n2,n1] = size(W20);
regc = beta/sqrt(n1+n2);
%
w1dim = n1*n0; w2dim = n2*n1;
n = w1dim + w2dim;
u0 = [reshape(W10,w1dim,1); reshape(W20,w2dim,1)];
%
% NOTE: Template below implies that target is X and regularization matrices
%  are set to the initial weights
cfun = @(u)FFN1Hidcost(u,X,X,n0,n1,n2,N,regc,W10,W20);
fu0 = cfun(u0);
%
opts = lbfgs('defaults');
opts.Display = 'off';
opts.MaxFuncEvals = 20000;
opts.MaxIters = 5000;
opts.StopTol = max(sqrt(eps),fu0*5.d-4/n); %NOTE: Cut of by sqrt(eps)
Out = Lbfgsc1(cfun,u0,opts);
u = Out.X;
fval = Out.F;
optits = Out.Iters;

%
% Computation of prediction error
W1 = reshape(u(1:w1dim),n1,n0);
W2 = reshape(u(w1dim+1:end),n2,n1);
%
XHid = W1*X';
F1 = 2./(1+exp(-2*XHid)) - 1; %Hidden layer output
XHid = XHid'; %(N,n1)
AEOut = (W2*F1)'; %(N,n2 = n0)
ae_err = mean(sqrt(sum((AEOut-X).^2,2))); %MRSE

if flag
    fprintf('Train1HL/Lbfgs: n= %5d, f0= %2.3e, it= %5d, f^*= %2.3e, TrE= %2.2e\n',...
        n,fu0,optits,fval,ae_err);
end

%
end
