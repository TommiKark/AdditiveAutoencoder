function [f,g] = FFN1Hidcost(u,X,Y,n0,n1,n2,N,regc,W1r,W2r)
%
% Computes regularized LSE + gradient for one-hidden-layer FFN.
% Author: Tommi Karkkainen (April, 2020)
%

%
% Restrore layerwise weight matrices from solution candidate vector u
%
w1dim = n1*n0; w2dim = n2*n1;
W1 = reshape(u(1:w1dim),n1,n0);
W2 = reshape(u(w1dim+1:end),n2,n1);
%
% Cost function and gradient matrices with matrix computations
%
% A. Output error and regularized LSE
W1X = W1*X'; %(n1,N)
F1 = 2./(1 + exp(-2*W1X)) - 1; D1 = 1 - F1.^2; %(n1,N)
E = W2*F1 - Y'; %(n2,N)
C = 1/N;
f = C/2*sum(E(:).^2) + regc/2*(sum(sum((W1-W1r).^2)) + sum(sum((W2-W2r).^2)));
%
% B. Derivative matrices for both layers
D1 = D1.*(W2'*E); %(n1,N)
DW1 = regc*(W1-W1r) + C*D1*X; %(n1,n0)
DW2 = regc*(W2-W2r) + C*E*F1'; %(n2,n1)
%
% Reshape as gradient vector
%
g = [reshape(DW1,w1dim,1); reshape(DW2,w2dim,1)];

%
end
