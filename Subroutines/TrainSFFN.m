function [W,ae_err,XHid] = TrainSFFN(X,beta,W0,Solver,FldLbs,ErrCof, ...
    MxIts,flag)
%
% Trains a 'symmetric' feedforward autoencoder by minimizing regularized LSE
% Author: Tommi Karkkainen (April, 2020)
%
% Inputs
% - X: training data
% - beta: regularization coefficient
% - W0: Initial weight values in a cell
% - Solver: Which solver to use ('Fminunc', 'Lbfgs', 'Adam', 'DsLbfgs', 'DsAdam')
% - FldLbs: Minibatch labels for the Ds-versions of the solvers
% - ErrCof: Multiplier for scaling the stopping criteria
% - flag: Flag for printing result information after optimization
%
% Outputs
% - W: Final weight matrices in a cell
% - ae_err: MRS autoencoding reconstruction error
% - fval: Final cost function value
% - XHid: Encoded data in the middle, squeezing layer

%
% Compute the total number of weight matrix rows to scale regularization.
% Reshape initial weights in the cell as one column vector.
%
nW = length(W0); %Number of weight matrices
nWrows = 0;
Ldims = zeros(1,nW+1); %Layer dimensions from input layer to middle, squeezing layer
u0 = [];
for i=1:nW
    [n,m] = size(W0{i});
    nWrows = nWrows + n;
    Ldims(i) = m; Ldims(i+1) = n;
    u0 = [u0; reshape(W0{i},n*m,1)];
end
n = length(u0);

%
% Define scaled regularization coefficient and function handle to the 
% function to be minimized.
% NOTE: Two versions for solvers with and without Ds, i.e., with and
%  without data matrix.
%
regc = beta/sqrt(nWrows);
Wr = W0; %Simply set the initial weight values as regularization matrices
cfun1 = @(u)FFNsymcost(u,X,regc,Wr); %Handle with weights as parameter
fu0 = cfun1(u0);
cfun2 = @(u,X)FFNsymcost(u,X,regc,Wr); %Handle with weights and data as parameters

if strcmp(Solver,'Fminunc')
    opts = optimset('fminunc'); %Default settings of FunEvals, Tolerances etc.
    opts.GradObj = 'on';
    opts.Display = 'off';
    opts.MaxIter = MxIts;
    [u,fval,~,Out] = fminunc(cfun1,u0,opts);
    optits = Out.iterations;
elseif strcmp(Solver,'Lbfgs')
    opts = lbfgs('defaults');
    opts.Display = 'off';
    opts.MaxIters = MxIts;
    opts.MaxFuncEvals = 20*MxIts; %According to default def of Lbfgs
    opts.StopTol = fu0*ErrCof/n;
    opts.RelFuncTol = -1.d-3; %Enforce this stopping criterion to be omitted
    %
    Out = Lbfgsc1(cfun1,u0,opts); %Version where all processing in one file
    u = Out.X;
    fval = Out.F;
    optits = Out.Iters;
elseif strcmp(Solver,'DsLbfgs')
    opts = lbfgs('defaults');
    opts.Display = 'off';
    opts.MaxIters = MxIts;
    opts.MaxFuncEvals = 20*MxIts; %According to default def of Lbfgs
    opts.StopTol = fu0*ErrCof/n;
    opts.RelFuncTol = -1.d-3;
    %
    Out = DsLbfgsc1(cfun2,u0,X,FldLbs,opts);
    u = Out.X;
    fval = cfun2(u,X);
    optits = Out.Iters;
elseif strcmp(Solver,'Adam')
    StopTol = ErrCof;
    MaxFunEvals = MxIts;
    [u,fval,optits] = own_adam(cfun1,u0,StopTol,MaxFunEvals);
elseif strcmp(Solver,'DsAdam') %Adam with Distributionally optimal sampling
    StopTol = ErrCof;
    MaxFunEvals = MxIts;
    [u,fval,optits] = own_dsadam(cfun2,u0,X,FldLbs,StopTol,MaxFunEvals);
    fval = cfun2(u,X);
else
    disp('Not supported solver type in TrainSFFN. Returning...')
    W = W0;
    ae_err = realmax;
    XHid = [];
    return
end

%
% MRS autoencoding error.
%
[W,ae_err,XHid] = AEMRSE(X,u,Ldims);

if flag
    fprintf('TrainSFFN/%s: n= %5d, f0= %2.3e, it= %5d, f^*= %2.3e, TrE= %2.2e\n',...
        Solver,n,fu0,optits,fval,ae_err);
end

%
end
