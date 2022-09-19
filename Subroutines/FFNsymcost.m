function [f,g] = FFNsymcost(u,X,regc,Wr)
%
% Regularized LSE cost and sensitivity for 'symmetric' FFN of the form 
% (example below with seven hidden layers)
% W^1' F(W^2' F(W^3' F(W^4' F(W^4 F(W^3 F(W^2 F(W^1 x))))))))
% Author: Tommi Karkkainen (April, 2020)
%
% Input
% - u: vector where all weight matrices are stored
% - X: input data
% - regc: regularization coefficient
% - Wr: target matrices for regularization in a cell 
% Output
% - f: Cost function value at u
% - g: gradient vector at u

%
% Restore matrices from the solution vector
%
nW = length(Wr); %Number of weight matrices
Ldims = zeros(1,nW+1); %Layer dimensions from input layer to middle, squeezing layer
W = {};
lastind = 0;
firstind = 1;
for i=1:nW
    [n,m] = size(Wr{i});
    Ldims(i) = m; Ldims(i+1) = n;
    lastind = lastind + n*m;
    W{end+1} = reshape(u(firstind:lastind),n,m);
    firstind = lastind + 1;
end

%
% Forward loop(s) for output error computation
%
F = {}; %Layerwise outputs
D = {}; %Layerwise derivatives
TmpMat = X';
for i=1:nW
    F{end+1} = 2./(1 + exp(-2*W{i}*TmpMat)) - 1; %tanh as activation function
    D{end+1} = 1 - F{end}.^2;
    TmpMat = F{end};
end
for i=nW:-1:2
    F{end+1} = 2./(1 + exp(-2*W{i}'*TmpMat)) - 1;
    D{end+1} = 1 - F{end}.^2;
    TmpMat = F{end};
end
E = W{1}'*TmpMat - X';
%
% Cost function and derivate matrices wrt regularization terms
%
N = size(X,1);
C = 1/N;
f = C/2*sum(E(:).^2);
regc2 = regc/2;
DW = {};
for i=1:nW
    TmpMat = W{i} - Wr{i};
    f = f + regc2*sum(TmpMat(:).^2);
    DW{end+1} = regc*TmpMat;
end

if nargout > 1 %Also gradient required by the calling program

    %
    % Backpropagation in matrix form
    % NOTE: Overwrites layerwise derivative matrices for reduced storage.
    %
    TmpMat = E;
    firstind = 1;
    for i=length(D):-1:nW
        D{i} = D{i}.*(W{firstind}*TmpMat);
        TmpMat = D{i};
        firstind = firstind + 1;
    end
    lastind = nW;
    for i=nW-1:-1:1
        D{i} = D{i}.*(W{lastind}'*TmpMat);
        TmpMat = D{i};
        lastind = lastind - 1;
    end
    %
    % Compute final derivative matrices and reshape as gradient vector
    %
    DW{1} = DW{1} + C*(D{1}*X + F{end}*E');
    g = reshape(DW{1},Ldims(2)*Ldims(1),1);
    lastind = length(F);
    for i=2:nW
        DW{i} = DW{i} + C*(D{i}*F{i-1}' + F{lastind-1}*D{lastind}');
        g = [g; reshape(DW{i},Ldims(i+1)*Ldims(i),1)];
        lastind = lastind - 1;
    end
end

%
end
