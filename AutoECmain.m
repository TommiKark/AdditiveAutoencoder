% 
% Main file for training an additive autoencoder with separate linear operator.
% Author: Tommi Karkkainen, University of Jyvaskyla (July, 2021)
%
% NOTE: The autoencoding approach
% 1) Prescale data to zero mean and range 2 featurewise to approximate bias
% 2) Use PCA to approximate linear behavior
% 3) Use feedforward autoendocer for the residual, to approximate nonlinear behavior
%
clear; close all;
%
% Include paths to all other directories with relevant code.
%
addpath Subroutines
addpath Subroutines/Adam
addpath Subroutines/Lbfgs
%
% Define flag that determines whether the results are illustrated or not.
%
DrawFlag = 1; %0 <> Plot results, 0 = No illustrations.
%
% There is a separate loading function that loads an individual test
% dataset after its abbreviation has been given below.
%
% Demonstration dataset
DataName = 'Iris'
%
% Small-dimension datasets
% DataName = 'Glass' %Nbr 1
% DataName = 'Wine' %Nbr 2
% DataName = 'Letter' %Nbr 3
% DataName = 'SML2010' %Nbr 4
% DataName = 'FrogMFCCs' %Nbr 5
% DataName = 'SteelPlates' %Nbr 6
% DataName = 'BreastCancerW' %Nbr 7
% DataName = 'Ionosphere' %Nbr 8
% DataName = 'Satimage' %Nbr 9
% DataName = 'SuperCond' %Nbr 10
% DataName = 'COIL2000' %Nbr 11
%
% Large-dimension datasets
% DataName = 'USPS' %Nbr 1
% DataName = 'BlogPosts' %Nbr 2
% DataName = 'CTSlices' %Nbr 3
% DataName = 'UJIIndoorLoc' %Nbr 4
% DataName = 'Madelon' %Nbr 5
% DataName = 'HumActRecog' %Nbr 6
% DataName = 'Isolet' %Nbr 7
% DataName = 'MNIST' %Nbr 8
% DataName = 'FashionMNIST' %Nbr 9
% DataName = 'COIL100' %Nbr 10
%
Data = LoadDataset(DataName); %Load Data matrix
%
% Supported five FFN structures: '1sym', '1hid', '3sym', '5sym', '7sym'
%
% Data structure to collect autoencoding errors for the tested set of
% shrinkage dimensions nredstart:nredstep:nredfin
%
nredstart = 1; %Smallest dimension of the middle, shrinkage layer
%nredstart = 10; %Smallest dimension of the middle, shrinkage layer
%nredfin = size(Data,2);%For Iris demonstration the actual data dimension
nredfin = size(Data,2) - 1; %Largest dimension of shrinkage layer: for small-dimension datasets up to next to the last
%nredfin = floor(0.6*size(Data,2)); %According to results, enough for determining the hidden dimension
nredstep = 1; %Increment of tested sizes of the scrinkage layer
%nredstep = 10; %Increment of tested sizes of the scrinkage layer
%
HidDims = nredstart:nredstep:nredfin;
NAEErrs = length(HidDims);
AEErrs = zeros(5,NAEErrs);
%
% Define the coefficients for the sizes of 7Sym's hidden layers. 3Sym uses
% the first one of these and 5Sym first and last.
%
HDimCoefs = [2 3 4]; %The default choice of the article
%HDimCoefs = [3 5 7]; %For testing whether changes results (btw, did not)
%
% Prescaling of data to zero mean and featurewise range 2
%
[N,n0] = size(Data);
m = mean(Data);
X = bsxfun(@minus,Data,m); %mean zero
minD = min(Data); maxD = max(Data);
if (min(maxD-minD) < sqrt(eps))
    disp('Constant variables in Data. Should be removed. Terminating.')
    return
end
cofs = 2./(maxD-minD); X = bsxfun(@times,X,cofs);  %range two featurewise
%
% PCA for the linear part.
%
[U,Y] = pca(X); %NOTE: Y = X*U
% If one wants to test the classical AE then simply set/uncomment 
%U = zeros(size(U));
%
% Compute the PC-AE errors.
%
PCAErr = zeros(1,NAEErrs);
for i = 1:NAEErrs
    nred = HidDims(i);
    PCArecon = Y(:,1:nred)*U(:,1:nred)'; %Enough for zero mean data
    PCAErr(i) = mean(sqrt(sum((X-PCArecon).^2,2))); %MRSE
end
%
% Define metaparameters.
%
%Regularization coefficient, fixed to a small value to slightly improve local coercivity
beta = 1.d-6; %close all;
%beta = 0.d0;
%How many restarts in optimization
% Two attempts seems enough for proper exploration
% But when searched through multiple hidden dimensions and well-aligned 
% pretrained models even one provides the qualitative result profile
inits = 1; 
%
% Supported solvers: 'Fminunc', 'Lbfgs', 'DsLbfgs', 'Adam', 'DsAdam'
%
% Define what optimization solver is used in pretraining and what in
% finetuning. Set the number of minibatches for both (applies only if Ds-solvers
% are used; Ds ~ Distribution sampling). Define also coefficients related to 
% stopping criteria and the maximun numbers of optimization iterations.
%
PreTrainer = 'DsLbfgs'; PTNMinib = 2; %Default choices used throughout
%PreTrainer = 'DsAdam'; PTNMinib = 9;
PTAcc = 1.d-5; PTMxIts = 2000;
FineTuner = 'Lbfgs'; FTNMinib = 1; %NOTE: Selection of 'Lbfgs' makes FTNMinib obsolete
FTAcc = 1.d-6; FTMxIts = 2000;
%FineTuner = 'DsAdam'; FTNMinib = 8;
%
% Define the folder where data structures related to results are being
% stored. NOTE: If empty then nothing stored.
%
ResDir = '';
%ResDir = './TestResults/.';
%
% Create labels for the folds of data for the minibatches in ds-versions.
%
PTMinibLbs = DOpLabelling(X,PTNMinib); %This takes time, be patient...
if FTNMinib == PTNMinib
    FTMinibLbs = PTMinibLbs; %Use the same if same number of folds
else
    FTMinibLbs = DOpLabelling(X,FTNMinib);
end
%
% Display the defined setting (NOTE: the last two prints test the foldings).
%
fprintf('AutoECmain: Dataset %s with N = %5d, n = %3d, inits = %1d.\n',...
    DataName,N,n0,inits);
fprintf('Pretrainer %s: Acc = %1.1e, MxIts = %5d, nbr of folds %2d (%2d/%2d).\n',...
    PreTrainer,PTAcc,PTMxIts,PTNMinib,min(PTMinibLbs),max(PTMinibLbs));
fprintf('Finetuner %s: Acc = %1.1e, MxIts = %5d, nbr of folds %2d (%2d/%2d).\n',...
    FineTuner,FTAcc,FTMxIts,FTNMinib,min(FTMinibLbs),max(FTMinibLbs));
%
% Create the storage directory if it does not exist, and default name for storage file.
%
if ~isempty(ResDir)
    SavFilNam = strcat(ResDir,DataName,'.mat')
    if ~exist(ResDir, 'dir')
        mkdir(ResDir)
    end
end
%
tic %Starts timer
for i = 1:NAEErrs
    nred = HidDims(i);
    fprintf('\nIter %2d: nred = %4d (of %4d)\n',i,nred,HidDims(end));
    PCArecon = Y(:,1:nred)*U(:,1:nred)';
    Xrest = X - PCArecon;

    % Do inits local searches of different residual models and store the most accurate.
    BestErrs = inf(5,1);
    for init=1:inits
        %
        %1Sym, nred-n0 (size(s) of scrink hidden layer(s))
        method = 1;
        W0 = 0.1*(2*rand(nred,n0) - 1); %Uniformly random initialization
        [W1c,ae_err] = TrainSFFN(Xrest,beta,{W0},PreTrainer,PTMinibLbs, ...
            PTAcc,PTMxIts,1);
        if (ae_err < BestErrs(method))
            W1Sym = W1c;
            BestErrs(method) = ae_err;
        end
        %
        %1Full, initialized from the result of 1sym
        method = 2;
        W0 = W1c{1};
        [W1,W2,ae_err] = Train1HidFFN(Xrest,beta,W0,W0',1);
        if (ae_err < BestErrs(method))
            W1Best = W1;
            W2Best = W2;
            BestErrs(method) = ae_err;
        end
        %
        %3Sym of size nred-HDimCoefs(1)-n0, stacked pretraining
        method = 3;
        n1 = round(HDimCoefs(1)*nred);
        W0 = 0.1*(2*rand(n1,n0) - 1);
        [W1,~,XHid] = TrainSFFN(Xrest,beta,{W0},PreTrainer,PTMinibLbs, ...
            PTAcc,PTMxIts,0);
        W0 = 0.1*(2*rand(nred,n1) - 1);
        W2 = TrainSFFN(XHid,beta,{W0},PreTrainer,PTMinibLbs,PTAcc,PTMxIts,0);
        [W3c,ae_err] = TrainSFFN(Xrest,beta,{W1{1},W2{1}},FineTuner, ...
            FTMinibLbs,FTAcc,FTMxIts,1);
        if (ae_err < BestErrs(method))
            W3Sym = W3c;
            BestErrs(method) = ae_err;
        end
        %
        %5Sym of size nred-HDimCoefs(1)-HDimCoefs(end)-n0, stacked pretraining
        method = 4;
        n1 = round(HDimCoefs(end)*nred);
        W10 = 0.1*(2*rand(n1,n0) - 1);
        [W1,~,XHid] = TrainSFFN(Xrest,beta,{W10},PreTrainer,PTMinibLbs, ...
            PTAcc,PTMxIts,0);
        n2 = round(HDimCoefs(1)*nred);
        W20 = 0.1*(2*rand(n2,n1) - 1);
        [W2,~,XHid] = TrainSFFN(XHid,beta,{W20},PreTrainer,PTMinibLbs, ...
            PTAcc,PTMxIts,0);
        W30 = 0.1*(2*rand(nred,n2) - 1);
        W3 = TrainSFFN(XHid,beta,{W30},PreTrainer,PTMinibLbs,PTAcc,PTMxIts,0);
        [W5c,ae_err] = TrainSFFN(Xrest,beta,{W1{1},W2{1},W3{1}},FineTuner,...
            FTMinibLbs,FTAcc,FTMxIts,1);
        if (ae_err < BestErrs(method))
            W5Sym = W5c;
            BestErrs(method) = ae_err;
        end
        %
        %7Sym of size nred-2*nred-3*nred-4*nred-n0, stacked pretraining
        method = 5;
        n1 = round(HDimCoefs(end)*nred);
        W10 = 0.1*(2*rand(n1,n0) - 1);
        [W1,~,XHid] = TrainSFFN(Xrest,beta,{W10},PreTrainer,PTMinibLbs, ...
            PTAcc,PTMxIts,0);
        n2 = round(HDimCoefs(2)*nred);
        W20 = 0.1*(2*rand(n2,n1) - 1);
        [W2,~,XHid] = TrainSFFN(XHid,beta,{W20},PreTrainer,PTMinibLbs, ...
            PTAcc,PTMxIts,0);
        n3 = round(HDimCoefs(1)*nred);
        W30 = 0.1*(2*rand(n3,n2) - 1);
        [W3,~,XHid] = TrainSFFN(XHid,beta,{W30},PreTrainer,PTMinibLbs, ...
            PTAcc,PTMxIts,0);
        W40 = 0.1*(2*rand(nred,n3) - 1);
        W4 = TrainSFFN(XHid,beta,{W40},PreTrainer,PTMinibLbs,PTAcc,PTMxIts,0);
        %
        [W7c,ae_err] = TrainSFFN(Xrest,beta,{W1{1},W2{1},W3{1},W4{1}}, ...
            FineTuner,FTMinibLbs,FTAcc,FTMxIts,1);
        if (ae_err < BestErrs(method))
            W7Sym = W7c;
            BestErrs(method) = ae_err;
        end
    end
    %
    % NOTE: Computations of reconstruction error below step-by-step to see
    % the patterns.
    %
    %1Sym autoencoding error starting from the normalized data
    method = 1;
    W1PCA = W1Sym{1}*(eye(n0) - U(:,1:nred)*U(:,1:nred)'); %Hidden with PCA
    AEOut = W1Sym{1}'*(2./(1+exp(-2*W1PCA*X')) - 1); %Autoencoder's output
    AEErrs(method,i) = mean(sqrt(sum((AEOut'-Xrest).^2,2))); %MRSE
    %1Hid error
    method = 2;
    W1PCA = W1Best*(eye(n0) - U(:,1:nred)*U(:,1:nred)'); %Hidden with PCA
    AEOut = W2Best*(2./(1+exp(-2*W1PCA*X')) - 1); %Autoencoder's output
    AEErrs(method,i) = mean(sqrt(sum((AEOut'-Xrest).^2,2))); %MRSE
    %3Sym error
    method = 3;
    W1PCA = W3Sym{1}*(eye(n0) - U(:,1:nred)*U(:,1:nred)'); %Hidden with PCA
    F = 2./(1+exp(-2*W1PCA*X')) - 1;
    F = 2./(1+exp(-2*W3Sym{2}*F)) - 1;
    F = 2./(1+exp(-2*W3Sym{2}'*F)) - 1;
    AEOut = W3Sym{1}'*F; %Autoencoder's output
    AEErrs(method,i) = mean(sqrt(sum((AEOut'-Xrest).^2,2))); %MRSE
    %5Sym error
    method = 4;
    W1PCA = W5Sym{1}*(eye(n0) - U(:,1:nred)*U(:,1:nred)'); %Hidden with PCA
    F = 2./(1+exp(-2*W1PCA*X')) - 1;
    for j=2:length(W5Sym)
        F = 2./(1+exp(-2*W5Sym{j}*F)) - 1;
    end
    for j=length(W5Sym):-1:2
        F = 2./(1+exp(-2*W5Sym{j}'*F)) - 1;
    end
    AEOut = W5Sym{1}'*F; %Autoencoder's output
    AEErrs(method,i) = mean(sqrt(sum((AEOut'-Xrest).^2,2))); %MRSE
    %7Sym error
    method = 5;
    W1PCA = W7Sym{1}*(eye(n0) - U(:,1:nred)*U(:,1:nred)'); %Hidden with PCA
    F = 2./(1+exp(-2*W1PCA*X')) - 1;
    for j=2:length(W7Sym)
        F = 2./(1+exp(-2*W7Sym{j}*F)) - 1;
    end
    for j=length(W7Sym):-1:2
        F = 2./(1+exp(-2*W7Sym{j}'*F)) - 1;
    end
    AEOut = W7Sym{1}'*F; %Autoencoder's output
    AEErrs(method,i) = mean(sqrt(sum((AEOut'-Xrest).^2,2))); %MRSE
    %
    if ~isempty(ResDir)
        save(SavFilNam, 'DataName', 'N', 'n0', 'inits', 'nred', 'PCAErr',...
            'AEErrs', 'PreTrainer', 'FineTuner', 'PTNMinib', 'FTNMinib',...
            'PTAcc', 'PTMxIts', 'FTAcc', 'FTMxIts', 'nredstart' , ...
            'nredstep', 'nredfin', 'HidDims');
    end
    toc
end

if DrawFlag
    %
    % Qualitatively separating colors according to
    %  https://colorbrewer2.org/#type=qualitative&scheme=Accent&n=6
    %
    CPalette = [166 206 227; 31 120 180; 178 223 138; 51 160 44; ...
        251 154 153; 227 26 28];
    CPalette = CPalette/255;
    %
    % Plotting style definitions.
    %
    LW = 2.5;
    MS = 13;
    FS = 22;
    %
    figure('Position',[625 160 910 620])
    % Plot first the PCA result
    Clr = CPalette(1,:);
    plot(HidDims,PCAErr,'--*','Color',Clr,'MarkerFaceColor',Clr,'LineWidth', ...
        LW,'MarkerSize',MS); hold on
    % Add results from nonlinear models
    for i =1:5
        Clr = CPalette(i+1,:);
        plot(HidDims,AEErrs(i,:),'--*','Color',Clr,'MarkerFaceColor', ...
            Clr,'LineWidth',LW,'MarkerSize',MS);
        hold on
    end
    legend('PCA','1Sym','1Hid','3Sym','5Sym','7Sym')
    TitStr = strcat('MRSE for',{' '},DataName);
    title(TitStr)
    set(gca,'FontSize',FS,'XLim',[HidDims(1)-nredstep HidDims(end)+nredstep]);
    xlabel('SqDim')
    ylabel('RMSE')
    grid on
end
