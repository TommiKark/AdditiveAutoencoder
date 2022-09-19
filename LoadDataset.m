function Data = LoadDataset(DataName)
%
% Loads dataset defined by DataName.
% Provides also links to the original sources of datasets.
% Author: Tommi Karkkainen (June, 2020)
%

switch DataName
    case 'Iris' %Test dataset
        % Iris in UCI: https://archive.ics.uci.edu/ml/datasets/iris
        % Iris directly from Matlab: N = 150, n = 4
        load fisheriris
        Data = meas;
    case 'Glass' %Smaller dataset 1
        % Glass from UCI: N = 214, n = 9 (running index in the first column)
        % https://archive.ics.uci.edu/ml/datasets/glass+identification
        load('./SmallerDatasets/glass.dat')
        Data = glass(:,2:end);
    case 'Wine' %Smaller dataset 2
        % Wine from UCI: N = 178, n = 13 (class label in the first column)
        % http://archive.ics.uci.edu/ml/datasets/Wine/
        load('./SmallerDatasets/wine.dat')
        Data = wine(:,2:end);
    case 'Letter' %Smaller dataset 3
        % Letter recognition from UCI: N = 20 000, n = 16
        % https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
        Data = importdata('./SmallerDatasets/LetterRecognition/letter-recognition.data');
        Data = Data.data;
    case 'SML2010' %Smaller dataset 4
        % N = 2763, n0 = 17
        % http://archive.ics.uci.edu/ml/datasets/SML2010
        Data = importdata('./SmallerDatasets/SML2010.txt',' ',1);
        Data = Data.data(:,3:end); %Columns 1-2 provide date and time
        I = find(max(Data) - min(Data) > sqrt(eps)); %Nonconstant variables
        Data = Data(:,I);
    case 'FrogMFCCs' %Smaller dataset 5
        % MFCCs from UCI: N = 7195, n = 22
        % http://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29
        Data = xlsread('./SmallerDatasets/Frogs_MFCCs.csv','A2:V7196');
        % For repetitive use more efficient to save and load in mat format.
        %save('./SmallerDatasets/FrogsMFCCs.mat', 'Data')
        %load ./SmallerDatasets/FrogsMFCCs
    case 'SteelPlates' %Smaller dataset 6
        % Steel plates from UCI: N = 1941, n = 27 (Class encoding in columns 28-34)
        % https://archive.ics.uci.edu/ml/datasets/Steel+Plates+Faults
        Data = dlmread('./SmallerDatasets/steel_plates.dat','\t');
        Data = Data(:,1:27);
    case 'BreastCancerW' %Smaller dataset 7
        % Breast Cancer Wisconsin (Diagnostic) Data Set
        % http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
        % N = 569, n = 30 (ID and class labels omitted from columns 1-2)
        Data = importdata('./SmallerDatasets/wdbc.data');
        Data = Data.data;
    case 'Ionosphere' %Smaller dataset 8
        % Matlab's sample dataset Ionosphere
        % N = 351, n = 33 (after removing constant variables)
        % UCI: https://archive.ics.uci.edu/ml/datasets/Ionosphere
        load ionosphere
        I = find(max(X) - min(X) > sqrt(eps)); %Nonconstant variables
        Data = X(:,I);
    case 'Satimage' %Smaller dataset 9
        % Statlog: Satellite images datas from UCI: N = 6435, n = 36
        % https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/
        % Train and test files combined as one dataset
        load('./SmallerDatasets/statlog_satimage_train.dat')
        Data = statlog_satimage_train(:,1:end-1); %Last column class variable
        load('./SmallerDatasets/statlog_satimage_test.dat')
        Data = [Data; statlog_satimage_test(:,1:end-1)]; %Last column class variable
    case 'SuperCond' %Smaller dataset 10
        % Superconductivity from UCI: N = 21 263, n = 82
        % http://archive.ics.uci.edu/ml/datasets/Superconductivty+Data 
        Data = xlsread('./SmallerDatasets/SuperCond/train.csv','A2:CD21264');
        % For repetitive use more efficient to save and load in mat format.
        %save('./SmallerDatasets/SuperCondTrain.mat', 'Data')
        %load ./SmallerDatasets/SuperCondTrain
    case 'COIL2000' %Smaller dataset 11
        % Insurance Company Benchmark from UCI: N = 5 822, n = 85
        %  http://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+%28COIL+2000%29
        Data = load('./SmallerDatasets/COIL2000/ticdata2000.txt');
        Data = Data(:,1:end-1); %Last column appear as binary class label
    case 'USPS'
        % USPS classic: N = 9 298, n = 256
        % From http://www.cad.zju.edu.cn/home/dengcai/Data/MLData.html
        load ./LargerDatasets/USPS/USPS;
        Data = fea;
    case 'BlogPosts'
        % Blog posts: N = 52 397, n = 277
        %  http://archive.ics.uci.edu/ml/datasets/BlogFeedback
        Data = xlsread('./LargerDatasets/BlogPosts/blogData_train.csv','A1:JU52397');
        I = find(max(Data) - min(Data) > sqrt(eps)); %Nonconstant variables
        Data = Data(:,I);
        % For repetitive use more efficient to save and load in mat format.
        %save('./LargerDatasets/BlogPosts/BlogDataTrain.mat', 'Data')
        %load ./LargerDatasets/BlogPosts/BlogDataTrain
    case 'CTSlices'
        % CT slices from UCI: N = 53 500, n = 379
        %  http://archive.ics.uci.edu/ml/datasets/Relative+location+of+CT+slices+on+axial+axis
        %  NOTE: First column 'patientid' left out. 
        Data = xlsread('./LargerDatasets/CTSlices/slice_localization_data.csv','B2:NU53501');
        I = find(max(Data) - min(Data) > sqrt(eps)); %Nonconstant variables
        Data = Data(:,I);
        % For repetitive use more efficient to save and load in mat format.
        %save('./LargerDatasets/CTSlices/SliceLocData.mat', 'Data')
        %load ./LargerDatasets/CTSlices/SliceLocData
    case 'UJIIndoorLoc'
        % Indoor localization from UCI: N = 19 937, n = 473
        %  https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc
        Data = xlsread('./LargerDatasets/UJIIndoorLoc/trainingData.csv','A2:TH19938');
        I = find(max(Data) - min(Data) > sqrt(eps)); %Nonconstant variables
        Data = Data(:,I);
        % For repetitive use more efficient to save and load in mat format.
        %save('./LargerDatasets/UJIIndoorLoc/UJIIndoorLocTrain.mat', 'Data')
        %load ./LargerDatasets/UJIIndoorLoc/UJIIndoorLocTrain
    case 'Madelon'
        % MADELON: artificial dataset for NIPS 2003 feature selection challenge
        % N (train) = 2000, N(test) = 1800, N(valid) = 600; n = 500
        %  http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/
        Data = load('./LargerDatasets/MADELON/madelon_train.data');
        data = load('./LargerDatasets/MADELON/madelon_test.data');
        Data = [Data; data];
        data = load('./LargerDatasets/MADELON/madelon_valid.data');
        Data = [Data; data];
    case 'HumActRecog'
        % Hum Act Recog: N = 7 351, n = 561
        % http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
        Data = importdata('./LargerDatasets/HumActRecog/X_train.txt',' ',1);
        Data = Data.data;
    case 'Isolet'
        % Isolet from UCI: N = 7 797, n = 617
        % http://archive.ics.uci.edu/ml/datasets/ISOLET
        load './LargerDatasets/Isolet/Isolet1'
        Data = fea;
        load './LargerDatasets/Isolet/Isolet2'
        Data = [Data; fea];
        load './LargerDatasets/Isolet/Isolet3'
        Data = [Data; fea];
        load './LargerDatasets/Isolet/Isolet4'
        Data = [Data; fea];
        load './LargerDatasets/Isolet/Isolet5'
        Data = [Data; fea];
    case 'MNIST'
        % MNIST classic: N = 60 000, n = 717
        % Raw: https://archive.ics.uci.edu/ml/machine-learning-databases/mnist-mld/
        % From http://www.cad.zju.edu.cn/home/dengcai/Data/MLData.html
        load ./LargerDatasets/MNIST/MNIST
        I = find(max(Data) - min(Data) > sqrt(eps)); %Nonconstant variables
        Data = Data(:,I);
    case 'FashionMNIST'
        % Fashion MNIST from GitHub of Salando: N = 60 000, n = 784
        % https://github.com/zalandoresearch/fashion-mnist
        load ./LargerDatasets/FashionMNIST/FashionMNIST
    case 'COIL100'
        % COIL100 classic: N = 7 200, n = 1024
        % From http://www.cad.zju.edu.cn/home/dengcai/Data/MLData.html
        load ./LargerDatasets/COIL100/COIL100
        Data = double(fea);
    otherwise
        fprintf('DataName %s not recognized in LoadDataset.\n',DataName);
        Data = [];
end

end
