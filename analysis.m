clear;
%% Load training.data
% Process the raw data, 
% convert each latter to a number based on ascii table
fileID = fopen('training.data');
C = textscan(fileID,'%d %s');
result = cell2mat(C(1));

features = [];
for k=1:length(C{2})
    s = C{2}{k};
    d = double(s(2:end)) - double('A');
    features = vertcat(features,d);
end
%% Start training, 
number_of_train = 2333;
% Separate all data to training and small portion of varify data
% To get a RMSE of the current model
train_result = result(1:number_of_train);
train_feature = features(1:number_of_train,:);
varify_result = result((number_of_train+1):end);
varify_feature = features((number_of_train+1):end,:);

Mdl = fitcecoc(features,result); 
% For multiple classes, use api: fitcecoc to use SVM

label = predict(Mdl, varify_feature);
RMSE = sqrt(mean((label - varify_result).^2)) 
% Showing the result of evaluation of this line
[c,cm,ind,per] = confusion(label,varify_result) 
% Showing the confusion matrix
%% Generating predictions from test.data using Mdl 
fileID = fopen('test.data');
C2 = textscan(fileID,'%s');
test_features = [];
for k=1:length(C2{1})
    s = C2{1}{k};
    d = double(s) - double('A');
    test_features = vertcat(test_features,d);
end
test_result = predict(Mdl, test_features);
%% Write the data to result.txt
dlmwrite('result.txt',test_result)
