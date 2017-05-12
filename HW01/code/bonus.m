clear all
D=importdata('hw1-scores.txt');
[row,col]=size(D);
index=zeros(row,col);
uni=zeros(row,1);
ran=zeros(1,col);
num=0;
for i=1:row
    [B,I]=sort(D(i,:),'descend');
    [C,I]=sort(I);
    index(i,:)=I;
end
D_sort=sort(D,2);
for i=1:row
    uni(i)=length(unique(D(i,:)));
end
for i=1:row
    if(uni(i)==5)
        for j=1:col
            temp=D(i,j);
            for k=1:col
                if D(i,k)==temp
                    index_sum=index(i,j)+index(i,k);
                    index(i,j)=index_sum./2;
                    index(i,k)=index_sum./2;
                end
            end
        end            
    end
end
for i=1:col
    ran(i)=mean(index(:,i));
end
