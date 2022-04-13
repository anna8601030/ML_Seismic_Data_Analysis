clear; clc; close all
region='WS';
component = 'Ecomp';
station = 'All-station';

% Data = sprintf('%s_%s_%s_feature.mat',Sta,year,date);
Data = sprintf('%s_%s_%s_feature.mat',region,component,station);
C_data = importdata(Data);

num_W = 15648 ;
num_C = 41262 ;
num_E = 10032 ;

% num_W = 0 ;
% num_C = 20000 ;
% num_E = 10000 ;

feature = 29;

data = [C_data{1}(1:num_W,:);C_data{2}(1:num_C,:);C_data{3}(1:num_E,:)];
label(1:num_W) = 1;
label(num_W+1:num_W+num_C) = 2;
label(num_W+num_C+1:num_W+num_C+num_E) = 3;
label = label';
data1 = [C_data{1}(1:num_W,:)];
data2 = [C_data{2}(1:num_C,:)];
data3 = [C_data{3}(1:num_E,:)];
%%
rowrank = randperm(size(data1, 1)); % 隨機打亂的數字，從1~行數打亂
rawdata1=data1(rowrank,:); %按照rowrank打亂矩陣的行數 B = A(rowrank,:)
class1=rawdata1 ;
%C2的資料 Tremor
% class2=load('ran_T_ELD');
%C3的資料 Noise
rowrank = randperm(size(data2, 1)); % 隨機打亂的數字，從1~行數打亂
rawdata2=data2(rowrank,:); %按照rowrank打亂矩陣的行數 B = A(rowrank,:)
class2=rawdata2 ;

rowrank = randperm(size(data3, 1)); % 隨機打亂的數字，從1~行數打亂
rawdata3=data3(rowrank,:); %按照rowrank打亂矩陣的行數 B = A(rowrank,:)
class3=rawdata3 ;

k1=1;
k2=5;
% N1=33;%%numbers of positive
N1=size(class1,1);
% N2=33;%numbers of negative
N2=size(class2,1);

N3=size(class3,1);
N4=N1+N2+N3 ;

%% 整個training data
all=[class1(1:N1,:);class2(1:N2,:);class3(1:N3,:)];
%% LOO
tic
%表示k從k1到k2,每兩個取一次（ex.1,3,5,7）
for k=k1:2:k2
    %做一個judge的矩陣，紀錄最後結果
    y=29;
    a=zeros(N4,3);
        for z=1:N1;
            %在C1資料內，選取一筆資料當testing data
            testinga=all(z,1:y);
            %把testing data從training data拿掉
            trainingB =all(:,1:y);
            trainingB(z,:) =[];
            for f=1:y;
                mC1(f)=mean(trainingB(:,f));
                devC1(f)=std(trainingB(:,f));
            end
            for ff=1:y;
                testing(ff)=[testinga(ff)-mC1(ff)]/devC1(ff);
            end
            size_train= size(trainingB,1);
            size_test=size(testing,1);
            trainingB=zscore(trainingB);
            %計算 testing 資料與每一個training data的距離
            for i = 1:size_test
                for j =1:size_train
                    distance(j,i)=norm(testing(i,:)-trainingB(j,:));
                end
            end
            [data, index] = sort(distance);
            %看k票，有幾票屬於C1 (若N1=100,index是1~99都是投給C1,因為有一筆資料是testing)
            c1_number= length(find(index(1:k,1)<N1));
            %有幾票屬於C2 (若index是100~199都是投給C2)
            c2_number= length(find(N2>index(1:k,1)>=N1));
            c3_number= length(find(index(1:k,1)>=N2));
            vote=[c1_number, c2_number, c3_number];
            [num, ii]=sort(vote,'descend');
            a(z,ii(1))=1;
            
            
        end
        for z=N1+1:N1+N2;
            %在C1資料內，選取一筆資料當testing data
            testinga=all(z,1:y);
            %把testing data從training data拿掉
            trainingB =all(:,1:y); trainingB(z,:) =[];
            for f=1:y;
                mC1(f)=mean(trainingB(:,f));
                devC1(f)=std(trainingB(:,f));
            end
            for ff=1:y;
                testing(ff)=[testinga(ff)-mC1(ff)]/devC1(ff);
            end
            size_train= size(trainingB,1);
            size_test=size(testing,1);
            trainingB=zscore(trainingB);
            %計算 testing 資料與每一個training data的距離
            for i = 1:size_test
                for j =1:size_train
                    distance(j,i)=norm(testing(i,:)-trainingB(j,:));
                end
            end
            [data, index] = sort(distance);
            %看k票，有幾票屬於C1 (若N1=100,index是1~99都是投給C1,因為有一筆資料是testing)
            c1_number= length(find(index(1:k,1)<N1));
            %有幾票屬於C2 (若index是100~199都是投給C2)
            c2_number= length(find(N2>index(1:k,1)>=N1));
            c3_number= length(find(index(1:k,1)>=N2));
            vote=[c1_number, c2_number, c3_number];
            [num, ii]=sort(vote,'descend');
            a(z,ii(1))=1;
            
        end
            
        for z=N2+1:N4;
            %在C1資料內，選取一筆資料當testing data
            testinga=all(z,1:y);
            %把testing data從training data拿掉
            trainingB =all(:,1:y); trainingB(z,:) =[];
            for f=1:y;
                mC1(f)=mean(trainingB(:,f));
                devC1(f)=std(trainingB(:,f));
            end
            for ff=1:y;
                testing(ff)=[testinga(ff)-mC1(ff)]/devC1(ff);
            end
            size_train= size(trainingB,1);
            size_test=size(testing,1);
            trainingB=zscore(trainingB);
            %計算 testing 資料與每一個training data的距離
            for i = 1:size_test
                for j =1:size_train
                    distance(j,i)=norm(testing(i,:)-trainingB(j,:));
                end
            end
            [data, index] = sort(distance);
            %看k票，有幾票屬於C1 (若N1=100,index是1~99都是投給C1,因為有一筆資料是testing)
            c1_number= length(find(index(1:k,1)<N1));
            %有幾票屬於C2 (若index是100~199都是投給C2)
            c2_number= length(find(N2>index(1:k,1)>=N1));
            c3_number= length(find(index(1:k,1)>=N2));
            vote=[c1_number, c2_number, c3_number];
            [num, ii]=sort(vote,'descend');
            a(z,ii(1))=1;
            
            
        end
        %%
        C11=sum(a(1:N1,1));
        C12=sum(a(1:N1,2));
        C13=sum(a(1:N1,3));
        C21=sum(a(N1+1:N1+N2,1));
        C22=sum(a(N1+1:N1+N2,2));
        C23=sum(a(N1+1:N1+N2,3));
        C31=sum(a(N2+1:N4,1));
        C32=sum(a(N2+1:N4,2));
        C33=sum(a(N2+1:N4,3));
        
        TPR1=(C11/N1)*100;
        TPR2=(C22/N2)*100;
        TPR3=(C33/N3)*100;
        
        TNR1=((C22+C33)/(C21+C31+C22+C33))*100;
        TNR2=((C11+C33)/(C12+C32+C11+C33))*100;
        TNR3=((C11+C22)/(C13+C23+C11+C22))*100;
        
        CR=((C11+C22+C33)/N4)*100;
        %ER=[1-((C11/N1)+(C22/N2))/2]*100;
        BA=(TPR1+TPR2+TPR3+TNR1+TNR2+TNR3)
        output(k,:)=[k,C11,C22,C33,TPR1,TPR2,TPR3,CR,BA];
       
end

output(find(sum(abs(output),2)==0),:)=[];

filename=sprintf('%s_%s_%s_knn.mat',region,component,station);
save(filename,'output')
toc
% %end