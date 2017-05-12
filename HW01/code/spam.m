clear all;
train=csvread('spambase_train.csv');
Mdl = fitcnb(train(:,1:57), train(:,58));
CVMdl = crossval(Mdl,'KFold',5);
L = kfoldLoss(CVMdl, 'lossfun', 'classiferror','mode','average');
fprintf('5-fold cross validation error is %f\n', L);
