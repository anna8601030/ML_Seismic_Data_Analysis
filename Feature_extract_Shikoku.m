clear; clc; close all
region='WS';
component = 'E';
station = 'All-station';

tic
for clas=1:3
    switch clas
        case 1
            
            [fn]=textread('/Users/anna/Desktop/Shikoku_project/List_Western_Shikoku/Tremor_List_E/Western/WS-w_20140820_20150329.E.tremor_list.txt','%s') ;
            
        case 2
            [fn]=textread('/Users/anna/Desktop/Shikoku_project/List_Western_Shikoku/Tremor_List_E/Central/WS-c_20140721_20150330.E.tremor_list.txt','%s');
            
        case 3
            [fn]=textread('/Users/anna/Desktop/Shikoku_project/List_Western_Shikoku/Tremor_List_E/Eastern/WS-e_20140716_20150214.E.tremor_list.txt','%s');
    end
    
    
    nfn=size(fn,1);                        %相當於自算有幾個檔案
    
    for ifn=1:nfn;                        %逐檔計算
        sacname=char(fn(ifn));
        [Ztime,Zdata,ZSAChdr] = fget_sac(sacname);
        waveform=Zdata;
        %waveform attributes
        meanseesee=mean(waveform);
        waveform=waveform-meanseesee;
        [upper,lower]=envelope(waveform);
        [a,b]=max(upper);
        tmax=(b-1)*0.01;
        myacf = acf(waveform,100);
        qq = abs(myacf);
        %0.1-1Hz
        databp1 = eqfiltfilt(waveform,0.1,1,0.01,4);
        [upper_bp1,lower_bp1]=envelope(databp1);
        %2-8Hz
        databp2 = eqfiltfilt(waveform,2,8,0.01,4);
        [upper_bp2,lower_bp2]=envelope(databp2);
        [a_bp2,b_bp2]=max(upper_bp2);
        %5-20Hz
        databp3 = eqfiltfilt(waveform,5,20,0.01,4);
        [upper_bp3,lower_bp3]=envelope(databp3);
        
        %%
        C_data{clas}(ifn,1)=mean(upper)/max(upper);
        C_data{clas}(ifn,2)=median(upper)/max(upper);
        C_data{clas}(ifn,3)=a;
        C_data{clas}(ifn,4)=tmax;
        C_data{clas}(ifn,5)=kurtosis(waveform);
        C_data{clas}(ifn,6)=kurtosis(upper);
        C_data{clas}(ifn,7)=skewness(waveform);
        C_data{clas}(ifn,8)=skewness(upper);
        C_data{clas}(ifn,9)=sum(qq(1:33,1));
        C_data{clas}(ifn,10)=sum(qq(34:100,1));
        C_data{clas}(ifn,11)=sum(qq(1:33,1))/sum(qq(34:100,1));
        C_data{clas}(ifn,12)=a_bp2;
        C_data{clas}(ifn,13)=length(findpeaks(upper_bp2));
        C_data{clas}(ifn,14)=sum(upper_bp1);
        C_data{clas}(ifn,15)=sum(upper_bp2);
        C_data{clas}(ifn,16)=sum(upper_bp3);
        C_data{clas}(ifn,17)=kurtosis(upper_bp1);
        C_data{clas}(ifn,18)=kurtosis(upper_bp2);
        C_data{clas}(ifn,19)=kurtosis(upper_bp3);
        
        [dftmax,dfts_max]=max(kurtosis(cwt(waveform)));
        dft=cwt(waveform);
        for tt=1:6000
            a(tt)=max(dft(:,tt))/mean(dft(:,tt));
            x(tt)=max(dft(:,tt))/median(dft(:,tt));
            c(tt)=max(dft(:,tt));
            d(tt)=mean(dft(:,tt));
            e(tt)=median(dft(:,tt));
        end
        C_data{clas}(ifn,20)=dftmax;
        C_data{clas}(ifn,21)=mean(a);
        %%
        C_data{clas}(ifn,22)=mean(x);
        
        [pks_max,locs_max] = findpeaks(abs(c));
        [pks_mean,locs_mean] = findpeaks(abs(d));
        [pks_median,locs_mean] = findpeaks(abs(e));
        
        C_data{clas}(ifn,23)=length(pks_max);
        C_data{clas}(ifn,24)=length(pks_mean);
        C_data{clas}(ifn,25)=length(pks_median);
        C_data{clas}(ifn,26)=length(pks_max)/length(pks_mean);
        C_data{clas}(ifn,27)=length(pks_max)/length(pks_median);
        %%
        [s,f,t] = spectrogram(waveform,50,20,6000,100);
        C_data{clas}(ifn,28)=sum(sum(s(122:300,:)))/sum(sum(s(302:1202,:)));
        C_data{clas}(ifn,29)=sum(sum(s(302:1202,:)))/sum(sum(s(301:3001,:)));
        
    end
end
% filename=sprintf('%s_%s_%s_feature.mat',station,component,day);
filename=sprintf('%s_%s_%s_feature.mat',region,component,station);
save(filename,'C_data')
toc