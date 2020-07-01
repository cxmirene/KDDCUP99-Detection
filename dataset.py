import pandas as pd
import numpy as np
from time import time

def handle():
    datafile = 'corrected.csv'
    savefile = 'corrected_save.csv'
    df = pd.read_csv(datafile, header=None)

    protocol = Protocol()
    df[1] = df[1].map(protocol)
    service = Service()
    df[2] = df[2].map(service)
    flag = Flag()
    df[3] = df[3].map(flag)
    label = Label()
    df[41] = df[41].map(label)

    symbolic = [1,2,3,6,11,13,14,20,21]         # 离散型特征的索引
    
    for j in range(df.shape[1]):
        if j in symbolic or j>=31:       # 排除离散型和后十个
            continue
        df_j_avg = df[j].mean()         # 均值
        df_j_mad = df[j].mad()          # 平均绝对偏差
        if df_j_avg==0 or df_j_mad==0:
            df[j]=0
            continue
        # 标准化
        df[j] = (df[j]-df_j_avg)/df_j_mad
        # 归一化
        df[j] = (df[j]-df[j].min())/(df[j].max() - df[j].min())
        print(str(j)+" 列处理完毕，剩余 "+str(df.shape[1]-j)+" 列未处理")

    print("均处理完毕，开始独热编码")

    columns_name = [str(i) for i in range(df.shape[1])]
    df.columns = columns_name
    symbolic_name = [str(i) for i in symbolic]
    df_result = pd.get_dummies(df, columns=symbolic_name)
    print("独热编码完毕")
    
    try:
        df_result.to_csv(savefile, index=None)
    except UnicodeEncodeError:
        print('写入错误')

def Protocol():
    protocol = {'tcp':0,'udp':1,'icmp':2}
    return protocol

def Service():
    list_ = ['aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard','domain','domain_u',
                 'echo','eco_i','ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest','hostnames',
                 'http','http_2784','http_443','http_8001','imap4','IRC','iso_tsap','klogin','kshell','ldap',
                 'link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat','nnsp','nntp',
                 'ntp_u','other','pm_dump','pop_2','pop_3','printer','private','red_i','remote_job','rje','shell',
                 'smtp','sql_net','ssh','sunrpc','supdup','systat','telnet','tftp_u','tim_i','time','urh_i','urp_i',
                 'uucp','uucp_path','vmnet','whois','X11','Z39_50']
    service = {}
    for i in range(len(list_)):
        service[list_[i]] = i
    return service

def Flag():
    list_ = ['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']
    flag = {}
    for i in range(len(list_)):
        flag[list_[i]] = i
    return flag

def Label():
    label_list = [
        ['normal.'],
        ['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.','apache2.','mailbomb.','processtable.','udpstorm.'],
        ['ipsweep.', 'nmap.', 'portsweep.', 'satan.','mscan.','saint.'],
        ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.','named.','sendmail.','snmpgetattack.','snmpguess.','warezmaster.','worm.','xlock.','xsnoop.'],
        ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.','httptunnel.','ps.','rootkit.','sqlattack.','xterm.']
    ]
    label = {}
    for i in range(len(label_list)):
        for j in range(len(label_list[i])):
            label[label_list[i][j]] = i
    return label

start = time()
handle()
end = time()
print("共耗时："+str(round((end-start)/60,3))+" min")
