function [W,ae_err,XHid,WNorms] = AEMRSE(X,u,Ldims)
% Computation of the MRS autoencoding error for the symmetric model.
% Author: T. Karkkainen (April, 2020)
%
nW = length(Ldims) - 1;
W = {};
lastind = 0;
firstind = 1;
WNorms = [];
for i=1:nW
    m = Ldims(i); n = Ldims(i+1);
    lastind = lastind + n*m;
    W{end+1} = reshape(u(firstind:lastind),n,m);
    firstind = lastind + 1;
    WNorms = [WNorms; norm(W{end},'fro')];
end
%
F = X';
for i=1:nW-1
    F = 2./(1 + exp(-2*W{i}*F)) - 1;
end
XHid = W{end}*F;
F = 2./(1 + exp(-2*XHid)) - 1;
XHid = XHid';
for i=nW:-1:2
    F = 2./(1 + exp(-2*W{i}'*F)) - 1;
end
AEOut = F'*W{1}; %(N,n0)
%
ae_err = mean(sqrt(sum((AEOut-X).^2,2))); %MRSE

%
end
