% Homework 04
% Support Vector Machines (Revisited)
% Author: Tianyang Chen

% import data
source_train = csvread('source_train.csv');
source_test = csvread('source_test.csv');
target_train = csvread('source_train.csv');
target_test = csvread('source_test.csv');

% rearrange data for matrix A, b, H, f
% X=[W1, W2, b, E1, ... E200]
[s_train_r, s_train_c]=size(source_train);
s_train_x=[source_train(:,1:s_train_c-1),ones(s_train_r,1)];
for i=1:s_train_r
    s_train_x(i,:)=s_train_x(i,:) .* source_train(i,s_train_c).*(-1);
end
s_tr_top=[s_train_x,(-1)*diag(ones(1,200))];
s_tr_bottom=[zeros(200,3),(-1)*diag(ones(1,200))];
A=[s_tr_top;s_tr_bottom];
b=[(-1)*ones(200,1);zeros(200,1)];
H=[diag([1,1]),zeros(2,201);zeros(201,203)];
f=[zeros(3,1);ones(200,1)];
x = quadprog(H,f,A,b);

% get W and b from source_train
w_s_train=x(1:2,1);
b_s_train=x(3,1);

% compute the accuracy for source_train
s_prediction2=zeros(200,1);
for i=1:200
    if source_train(i,1:2)*w_s_train+b_s_train<0
        s_prediction2(i)=-1;
    else
        if source_train(i,1:2)*w_s_train+b_s_train>0
            s_prediction2(i)=1;
        end
    end
end
correct_num2=0;
for i=1:200
    if s_prediction2(i)==source_train(i,3);
        correct_num2=correct_num2+1;
    end
end
fprintf('The accuracy of source_train is %f\n', correct_num2/200);

% compute the accuracy for source_test
s_prediction=zeros(400,1);
for i=1:400
    if source_test(i,1:2)*w_s_train+b_s_train<0
        s_prediction(i)=-1;
    else
        if source_test(i,1:2)*w_s_train+b_s_train>0
            s_prediction(i)=1;
        end
    end
end
correct_num=0;
for i=1:400
    if s_prediction(i)==source_test(i,3);
        correct_num=correct_num+1;
    end
end
fprintf('The accuracy of source_test is %f\n', correct_num/400);

% rearrange data for matrix At, bt, Ht, ft (here t denotes target)
% Xt=[W1, W2, b, E1, ... E200]
% B=C=1
t_train_x=[target_train(:,1:2),ones(200,1)];
for i=1:200
    t_train_x(i,:)=t_train_x(i,:) .* target_train(i,3) .* (-1);
end
t_tr_top=[t_train_x,(-1)*diag(ones(1,200))];
t_tr_bottom=[zeros(200,3),(-1)*diag(ones(1,200))];
At=[t_tr_top;t_tr_bottom];
bt=[(-1)*ones(200,1);zeros(200,1)];
Ht=[diag([1,1]),zeros(2,201);zeros(201,203)];
ft=[(-1)*w_s_train;0;ones(200,1)];
xt = quadprog(Ht,ft,At,bt);

% get target hyperplane from source hyperplane
w_t_train=xt(1:2,1);
b_t_train=xt(3,1);

% compute the accuracy for target_train
train_pre=zeros(200,1);
for i=1:200
    if target_train(i,1:2)*w_t_train+b_t_train<0
        train_pre(i)=-1;
    else
        if target_train(i,1:2)*w_t_train+b_t_train>0
            train_pre(i)=1;
        %else error('result is zero');
        end
    end
end
correct_target_train=0;
for i=1:200
    if train_pre(i)==target_train(i,3);
        correct_target_train=correct_target_train+1;
    end
end
fprintf('The accuracy of target_train is %f\n', correct_target_train/200);

% compute the accuracy for target_test
test_pre=zeros(400,1);
for i=1:400
    if target_test(i,1:2)*w_t_train+b_t_train<0
        test_pre(i)=-1;
    else
        if target_test(i,1:2)*w_t_train+b_t_train>0
            test_pre(i)=1;
        %else error('result is zero');
        end
    end
end
correct_target_test=0;
for i=1:400
    if test_pre(i)==target_test(i,3);
        correct_target_test=correct_target_test+1;
    end
end
fprintf('The accuracy of target_test is %f\n', correct_target_test/400);
        

            