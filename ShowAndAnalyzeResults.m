%
clear; close all;
%
Methods0 = {'PCA','1Sym','1Hid','3Sym','5Sym','7Sym'}
Methods = {'1Sym','1Hid','3Sym','5Sym','7Sym'}
%
% Qualitatively separating colors according to
%  https://colorbrewer2.org/#type=qualitative&scheme=Accent&n=6
%
CPalette = [166 206 227; 31 120 180; 178 223 138; 51 160 44; ...
    251 154 153; 227 26 28];
CPalette = CPalette/255;
%
ResultDir = './ASetOfResults/'
StoreDir = '' %If StoreDir empty then result figures not stored just shown
%StoreDir = './ResultStore/'

%
% Plotting style definitions.
%
LW = 2.5;
MS = 10;
FS = 22;

if ~isempty(StoreDir) && ~exist(StoreDir, 'dir')
  mkdir(StoreDir)
end

Files = dir(ResultDir);
I = find(contains({Files.name},'.mat'));
Files = Files(I);
NFiles = length(Files);

for i=1:NFiles

    file = strcat(ResultDir,char(Files(i).name));
    load(file)
    %
    fprintf('\nResults for %s: N = %d, n0 = %d, LastHdim = %d\n', ...
        DataName,N,n0,HidDims(end));
    fprintf(' inits %1d, PreTrainer %s, FineTuner %s, PTnfolds %2d, FTnfolds %2d.\n',...
        inits,PreTrainer,FineTuner,PTNMinib,FTNMinib);

    %
    % Threshold value for dimension estimation.
    %
    Thr = 3e-3; %One of two threshold values for large datasets
    
    AEEDiffs = abs(diff(AEErrs,1,2));
    MinDiffs = AEEDiffs(2,:); %Use 1Hid for hidden dimension
    hind = find(MinDiffs < Thr,1);
    if isempty(hind)
        Thr2 = 3e-2; %Second of two threshold values for large datasets
        hind = find(MinDiffs < Thr2,1);
        if isempty(hind)
            fprintf('Intrinsic dimension with threshold values %1.1e or %1.1e not detected.');
            error('Macro stop.')
        else
            fprintf('Threshold for intrinsic dimension changed from %1.1e to %1.1e.\n',...
                Thr,Thr2);
            Thr = Thr2;            
        end
    end
    hdim = HidDims(hind);
    MinAEErr = AEErrs(2,hind);
    fprintf('Suggested nonlinear data dimension = %3d (of %3d).\n',hdim,n0);
    fprintf('Thr %1.2e: Min-AE-Err = %1.2e.\n',Thr,MinAEErr);

    % 1st figure: MRSE of PCA, 1Sym and 1Hid
    Clr = CPalette(1,:);
    figure('Position',[625 160 910 620])
    plot(HidDims,PCAErr,'--*','Color',Clr,'MarkerFaceColor',Clr, ...
        'LineWidth',LW,'MarkerSize',MS);
    hold on
    %
    for c =1:2
        Clr = CPalette(c+1,:);
        plot(HidDims,AEErrs(c,:),'--*','Color',Clr,'MarkerFaceColor', ...
            Clr,'LineWidth',LW,'MarkerSize',MS);
        hold on
    end
    xlabel('SqDim'), ylabel('MRSE')
    xlim([-10 HidDims(end)+10])
    y = get(gca,'YLim'); y(1) = y(1) - 0.25; set(gca,'YLim',y);
    set(gca,'FontSize',FS); %,'XTick',HidDims(1:xInc:length(HidDims))) %,'YTick',[0 2 4])
    grid on
    %
    hold on, plot(hdim,MinAEErr,'or','MarkerSize',round(1.5*MS),...
        'LineWidth',round(1.5*LW))
    TitStr = strcat('MRSE for',{' '},DataName);
    title(TitStr)
    legend(Methods0(1:3))
    if ~isempty(StoreDir)
        Fname = strcat(StoreDir,DataName,'ShallowMRSE.fig'); saveas(gcf,Fname)
        Fname = strcat(StoreDir,DataName,'ShallowMRSE.png'); saveas(gcf,Fname)
    end
    
    % 2nd figure: MRSEDiff for 1Sym and 1Hid
    figure('Position',[625 160 910 620])
    %
    for c =1:2
        Clr = CPalette(c+1,:);
        plot(HidDims(2:end),AEEDiffs(c,:),'--*','Color',Clr,'MarkerFaceColor', ...
            Clr,'LineWidth',LW,'MarkerSize',MS);
        hold on
    end
    xlabel('SqDim'), ylabel('\Delta (MRSE)')
    xlim([HidDims(1) - 5 HidDims(end) + 5])
    y = get(gca,'YLim'); y(1) = -0.05; set(gca,'YLim',y);
    set(gca,'FontSize',FS); %,'XTick',HidDims(2:xInc:length(HidDims))) %,'YTick',[0 2 4])
    grid on
    legend(Methods(1:2),'AutoUpdate','off')
    %
    hold on, plot(HidDims(hind+1),MinDiffs(hind),'or','MarkerSize',round(1.5*MS),...
        'LineWidth',round(1.5*LW))
    hold on, plot(get(gca,'XLim'),[Thr Thr],'r--','LineWidth',round(1.5*LW))
    TitStr = strcat('MRSE improvement for',{' '},DataName);
    title(TitStr)
    if ~isempty(StoreDir)
        %disp('Finalize plot and then strike a key to store...'), pause
        Fname = strcat(StoreDir,DataName,'HidDimMRSE.fig'); saveas(gcf,Fname)
        Fname = strcat(StoreDir,DataName,'HidDimMRSE.png'); saveas(gcf,Fname)
    end
    
    % 3rd figure: Zoom of MRSEDiff for 1Sym and 1Hid
    figure('Position',[625 160 910 620])
    %
    for c =1:2
        Clr = CPalette(c+1,:);
        plot(HidDims(hind-3:end),AEEDiffs(c,hind-4:end),'--*', ...
            'Color',Clr,'MarkerFaceColor',Clr,'LineWidth',round(1.25*LW),...
            'MarkerSize',round(1.25*MS));
        hold on
    end
    xlabel('SqDim'), ylabel('\Delta (MRSE)')
    set(gca,'FontSize',FS)
    %y = get(gca,'YLim'); y(1) = -0.05; set(gca,'YLim',y);
    grid on
    legend(Methods(1:2),'AutoUpdate','off')
    %
    hold on, plot(HidDims(hind+1),MinDiffs(hind),'or','MarkerSize',round(1.5*MS),...
        'LineWidth',round(1.5*LW))
    hold on, plot(get(gca,'XLim'),[Thr Thr],'r--','LineWidth',round(1.5*LW))
    TitStr = strcat('Zoom of MRSE improvement for',{' '},DataName);
    title(TitStr)
    if ~isempty(StoreDir)
        %disp('Finalize plot and then strike a key to store...'), pause
        Fname = strcat(StoreDir,DataName,'ZoomHidDimMRSE.fig'); saveas(gcf,Fname)
        Fname = strcat(StoreDir,DataName,'ZoomHidDimMRSE.png'); saveas(gcf,Fname)
    end
    
    % 4th figure: MRSE for NN models
    figure('Position',[625 160 910 620])
    %
    for c =1:5
        Clr = CPalette(c+1,:);
        plot(HidDims,AEErrs(c,:),'--*','Color',Clr,'MarkerFaceColor', ...
            Clr,'LineWidth',LW,'MarkerSize',MS);
        hold on
    end
    xlabel('SqDim'), ylabel('MRSE')
    xlim([0 HidDims(end)+10])
    y = get(gca,'YLim'); y(1) = y(1) - 0.2; set(gca,'YLim',y);
    set(gca,'FontSize',FS); %,'XTick',HidDims(1:xInc:m)) %,'YTick',[0 2 4])
    grid on
    %
    hold on, plot(hdim,MinAEErr,'or','MarkerSize',round(1.5*MS),...
        'LineWidth',round(1.5*LW))
    TitStr = strcat('MRSE for',{' '},DataName);
    title(TitStr)
    legend(Methods)
    if ~isempty(StoreDir)
        %disp('Finalize plot and then strike a key to store...'), pause
        Fname = strcat(StoreDir,DataName,'MRSEDim.fig'); saveas(gcf,Fname)
        Fname = strcat(StoreDir,DataName,'MRSEDim.png'); saveas(gcf,Fname)
    end
    
    % 5th figure: Rel MRSE for NN models
    %RelAEErrs = (AEErrs(2,1:hind-1) - AEErrs(:,1:hind-1))./AEErrs(2,1:hind-1);
    %RelAEErrs = AEErrs(:,1:end)./AEErrs(2,1:end);
    RelAEErrs = AEErrs(:,1:hind-1)./AEErrs(2,1:hind-1);
    figure('Position',[625 160 910 620])
    for c =2:5
        Clr = CPalette(c+1,:);
        plot(HidDims(1:hind-1),RelAEErrs(c,1:hind-1),'--*','Color',Clr,...
            'MarkerFaceColor',Clr,'LineWidth',LW,'MarkerSize',MS);
        hold on
    end
    xlabel('SqDim'), ylabel('MRSE/1Hid')
    Y = [min(RelAEErrs(end,1:hind-1))-0.1 max(RelAEErrs(end,1:hind-1))+0.1];
    xlim([0 HidDims(hind)]), ylim(Y)
    set(gca,'FontSize',FS); %,'XTick',HidDims(1:xInc:m)) %,'YTick',[0 2 4])
    grid on
    %
    TitStr = strcat('Relative MRSE for',{' '},DataName);
    title(TitStr)
    legend(Methods(2:end),'Location','southwest')
    if ~isempty(StoreDir)
        %disp('Finalize plot and then strike a key to store...'), pause
        Fname = strcat(StoreDir,DataName,'MRSEH1Rel.fig'); saveas(gcf,Fname)
        Fname = strcat(StoreDir,DataName,'MRSEH1Rel.png'); saveas(gcf,Fname)
    end

    % Descriptive statistics for how much better a deeper model was during
    % search of a hidden dimension
    % NOTE: RelAEErrs say how much shallow model was improved. So the
    % reciprocal of this says how much better, i.e. more efficient, the
    % deeper model was?
    HDs = HidDims(1:hind-1);
    fprintf('Efficiency of Sym models for HidDims = %d-%d:\n',HDs(1),HDs(end));
    for c=[1 3:length(Methods)]
        InvRAEs = 1./RelAEErrs(c,:);
        MIRAE = mean(InvRAEs); [MaxIRAE,ind] = max(InvRAEs);
        %figure, plot(HDs,InvRAEs,'b--*')
        fprintf('Model %s: Mean eff. = %5.2f, Max eff. = %5.2f (hd = %d)\n',...
            Methods{c},MIRAE,MaxIRAE,HDs(ind));
    end
    
    disp('Strike a key...'), pause
    
    close all
  
end
