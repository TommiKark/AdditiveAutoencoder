function BatchLabels = DOpLabelling(Data,nbatches)
%
% Creating minibatches of data using Distribution optimal sampling.
% Author: Tommi Karkkainen (April, 2020)
%
numobs = size(Data,1);
if nbatches == 1 %Just copy and return
    BatchLabels = ones(numobs,1);
    return
end

BatchLabels = zeros(numobs,1);
Iv = [1:numobs]'; %Which index values to be distributed into nfolds
FoldLbs = [1:nbatches]';
while numobs >= nbatches
    x0 = Data(randperm(numobs,1),:);
    [~,I] = pdist2(Data(Iv,:),x0,'euclidean','Smallest',nbatches);
    I = Iv(I); %Original index values
    BatchLabels(I) = FoldLbs;
    Iv = setdiff(Iv,I);
    numobs = numobs - nbatches;
end

if numobs > 0
    BatchLabels(Iv) = [1:numobs]';
end

end
