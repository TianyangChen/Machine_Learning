% Homework 04
% An Experiment with Ensembles
% Author: Tianyang Chen

% import data
data = csvread('ionosphere.csv');
x=data(1:351,1:33);
y=data(1:351,34);

% Bagging
bagging=fitensemble(x,y,'Bag', 100, 'Tree','Type','classification');
bag_Loss = resubLoss(bagging,'Mode','Cumulative');

%Adaboosting
adaboost = fitensemble(x,y,'AdaBoostM1',100,'Tree');
boost_Loss = resubLoss(adaboost,'Mode','Cumulative');

%draw plot
plot(bag_Loss);
hold on 
plot(boost_Loss);
legend('bagging','adaboost')
xlabel('size of an ensemble');
ylabel('Testing Error');