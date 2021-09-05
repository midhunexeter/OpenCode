from Deep_Walker import deep_learner as deep
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.datasets import make_sparse_coded_signal
from scipy.sparse import rand
import scipy as sc
#Training a clssifier to detect different regions
from sklearn.svm import SVR,SVC,OneClassSVM
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier as rf_c
from sklearn.ensemble import RandomForestRegressor as rf_r
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture as GM
from sklearn.metrics import confusion_matrix
from sklearn import metrics as metr
from sklearn.multioutput import MultiOutputRegressor as multiout_r
from sklearn.ensemble import GradientBoostingRegressor
scaler = MinMaxScaler()
from sklearn.externals import joblib
import pywt

# -----------------------------------------------------------------------------
#Clustering to find out a better basis for the data

def optimizer_corr(exp):
    err_list = []
    cos_dist=[]
    corr_dist = []
    path_rel = 'the path'
    file_names = os.listdir(path_rel)
    fy_list=[]
    for fil in file_names:
        mod = np.load(path_rel+fil)
        
        x=mod[-1]
        th1=x[:,0]
        om1=x[:,2]
        dL=0.387
        mL=6.769
        
        Fy= dL*mL*(om1**2)*np.cos(th1)+(dL*mL*np.sin(th1)*om1)
        #Preprocessing
        mod =resample(sc.signal.detrend(Fy),int(np.shape(Fy)[0]/5))[400:800]
        #mod= sc.signal.detrend(mod,bp=[300,600,900,1200,1500,1700])
        mod = scaler.fit_transform(mod.reshape(-1,1))

        Signal_to_match = exp
        Signal_to_match = scaler.fit_transform(Signal_to_match.reshape(-1,1))
        #err_list.append(optimizer(mod,Signal_to_match))
        #cos_dist.append(spatial.distance.cosine(Signal_to_match,mod))
        lagsOut = np.arange(-len(Signal_to_match)+1, len(Signal_to_match))
        corrcoeflist = np.correlate(Signal_to_match[:,0],mod[:,0],'full')
        fy_list.append(mod)
        corr_dist.append(max(abs(corrcoeflist)))
    return [fy_list,fy_list[np.argmax(corr_dist)],np.argmax(corr_dist)]

def extract_cfs(inp,cfs):
    '''takes input a list or matrix of (somthing,150) and gives out coefficients in the weight bases from 'd' object'''
    
    
    weights_list=[]
    for weights in d.model_sl.layers[1].get_weights():  
        weights_list.append(weights)

    X  = np.array(weights_list)[0].T

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=cfs)
    coef_list =[]
    coef_ind_list = []
    coeff_ind_list_all=[]
    coef_ind_mat = np.zeros((cfs,np.shape(inp)[0]))
    i=0
    for y in inp:
        y = sc.signal.detrend(y)
        omp.fit(X, y)
        coef_list.append(omp.coef_)
        coef_ind_list.append(np.array(np.nonzero(omp.coef_)[0],dtype=int))
        coeff_ind_list_all.append(np.array(omp.coef_,dtype=int))
        
        if np.size(np.array(np.nonzero(omp.coef_)[0]))<cfs:
            print(np.size(np.array(np.nonzero(omp.coef_)[0])))
            temp = np.hstack((np.array(np.nonzero(omp.coef_)[0]),-1))
            while np.size(temp)<cfs:
                print('Appending -1s')
                temp = np.hstack((temp,-1))
        else:
            temp = np.array(np.nonzero(omp.coef_)[0])
        coef_ind_mat[:,i] = temp
        i=i+1
    
    #coef_ind_mat = np.array(coef_ind_list)
    coef_mat = np.array(coef_list,dtype=float)
    return[coef_ind_mat,coef_mat,coeff_ind_list_all]
    
    
def sparse_recon(inp,cfs):
    '''takes input a list or matrix of (somthing,150) and gives out coefficients in the weight bases from 'd' object'''
    
    
    weights_list=[]
    for weights in d.model_sl.layers[1].get_weights():  
        weights_list.append(weights)

    X  = np.array(weights_list)[0].T
    
    
    
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=cfs)
    omp_cflist = []
    y_list=[]
    for y in inp:
        y = sc.signal.detrend(y)
        y_list.append(y)
        omp.fit(X, y)
        omp_cflist.append(omp.coef_)

    return[omp_cflist,y_list]
    
    
    
def training_data(d):
    X = np.reshape(d.x_train,(d.x_train.shape[0],d.x_train.shape[1])).T
    y = d.y_train.T
    y_detr = sc.signal.detrend(y,axis=0)
    X_detr = sc.signal.detrend(X,axis=0)
    y_norm = np.abs(np.linalg.norm(y_detr,axis=0))
    input_norm_last_region = np.abs(np.linalg.norm(X_detr[-150:],axis=0))
    input_train = X_detr.T
    output_train = y_norm
    return [input_train,output_train,input_norm_last_region]


def training_data_classifier(d,thr):
    X = np.reshape(d.x_train,(d.x_train.shape[0],d.x_train.shape[1])).T
    y = d.y_train.T
    y_detr = sc.signal.detrend(y,axis=0)
    X_detr = sc.signal.detrend(X,axis=0)
    y_norm = np.abs(np.linalg.norm(y_detr,axis=0))
    input_norm_last_region = np.abs(np.linalg.norm(X_detr[-150:],axis=0))
    input_train = X_detr.T
    output_train = y_norm<thr
    return [input_train,output_train,input_norm_last_region]

def training_data_classifier_with_feature_extraction(d,thr):
    X = np.reshape(d.x_train,(d.x_train.shape[0],d.x_train.shape[1])).T
    y = d.y_train.T
    y_detr = sc.signal.detrend(y,axis=0)
    X_detr = sc.signal.detrend(X,axis=0)
    y_norm = np.abs(np.linalg.norm(y_detr,axis=0))
    input_norm_last_region = np.abs(np.linalg.norm(X_detr[-150:],axis=0))
    
    #feature_list = [ml.specgram(d.x_train[i,:][:,0],NFFT=100,noverlap=0)[0].T[:,0:1] for i in range(max(d.x_train.shape))]
    feature_list = [pywt.cwt(sc.signal.detrend(d.x_train[i,:][:,0]),2,'mexh')[0].T[0:400] for i in range(max(d.x_train.shape))]
    
    
    input_train = np.array(feature_list).reshape(np.array(feature_list).shape[0:2])
    output_train = y_norm<thr
    return [input_train,output_train,input_norm_last_region]


def training_data_all(d):
    X = np.reshape(d.x_train,(d.x_train.shape[0],d.x_train.shape[1])).T
    y = d.y_train.T
    y_detr = sc.signal.detrend(y,axis=0)
    X_detr = sc.signal.detrend(X,axis=0)
    #y_norm = np.abs(np.linalg.norm(y_detr,axis=0))
    input_norm_last_region = np.abs(np.linalg.norm(X_detr[-150:],axis=0))
    input_train = X_detr.T
    output_train = y_detr.T
    return [input_train,output_train,input_norm_last_region]
    
# Types of testing data
def testing_data(d):
    X = np.reshape(d.x_test,(d.x_test.shape[0],d.x_test.shape[1])).T
    y = d.y_test.T
    y_detr = sc.signal.detrend(y,axis=0)
    X_detr = sc.signal.detrend(X,axis=0)
    y_norm = np.abs(np.linalg.norm(y_detr,axis=0))
    input_norm_last_region = np.abs(np.linalg.norm(X_detr[-150:],axis=0))
    input_train = X_detr.T
    output_train = y_norm
    return [input_train,output_train,input_norm_last_region]

def testing_data_classifier(d,thr):
    X = np.reshape(d.x_test,(d.x_test.shape[0],d.x_test.shape[1])).T
    y = d.y_test.T
    y_detr = sc.signal.detrend(y,axis=0)
    X_detr = sc.signal.detrend(X,axis=0)
    y_norm = np.abs(np.linalg.norm(y_detr,axis=0))
    input_norm_last_region = np.abs(np.linalg.norm(X_detr[-150:],axis=0))
    input_train = X_detr.T
    output_train = y_norm<thr
    return [input_train,output_train,input_norm_last_region]

def testing_data_classifier_with_feature_extraction(d,thr):
    X = np.reshape(d.x_test,(d.x_test.shape[0],d.x_test.shape[1])).T
    y = d.y_test.T
    y_detr = sc.signal.detrend(y,axis=0)
    X_detr = sc.signal.detrend(X,axis=0)
    y_norm = np.abs(np.linalg.norm(y_detr,axis=0))
    input_norm_last_region = np.abs(np.linalg.norm(X_detr[-150:],axis=0))
    #feature_list = [ml.specgram(d.x_test[i,:][:,0],NFFT=100,noverlap=0)[0].T[:,0:1] for i in range(max(d.x_test.shape))]
    feature_list = [pywt.cwt(sc.signal.detrend(d.x_test[i,:][:,0]),2,'mexh')[0].T[0:400] for i in range(max(d.x_test.shape))]
    
    input_train = np.array(feature_list).reshape(np.array(feature_list).shape[0:2])
    output_train = y_norm<thr
    return [input_train,output_train,input_norm_last_region]


def testing_data_all(d):
    X = np.reshape(d.x_test,(d.x_test.shape[0],d.x_test.shape[1])).T
    y = d.y_test.T
    y_detr = sc.signal.detrend(y,axis=0)
    X_detr = sc.signal.detrend(X,axis=0)
    y_norm = np.abs(np.linalg.norm(y_detr,axis=0))
    input_norm_last_region = np.abs(np.linalg.norm(X_detr[-150:],axis=0))
    input_train = X_detr.T
    output_train = y_detr.T
    return [input_train,output_train,input_norm_last_region]

def plot_non_smooth(S=0,N=25):
    j=1
    for i in d.non_smooth_index[S:N]:
        plt.subplot(int(np.sqrt(N-S)),int(np.sqrt(N-S)),j)
        plt.plot(range(0,np.size(d.x_test[i])),d.x_test[i],'k')#Input
        plt.plot(range(np.size(d.x_test[i]),np.size(d.x_test[i])+np.size(d.y_test[i])),d.y_test[i],'g')#Original
        plt.plot(range(np.size(d.x_test[i]),np.size(d.x_test[i])+np.size(d.y_test[i])),d.y_pred_sl[i],'r')#Predictions
        #plt.plot(range(np.size(d.x_test[i]),np.size(d.x_test[i])+np.size(d.y_test[i])),clf.predict(input_test)[i],'r')#Predictions
        j=j+1
    plt.show()



def plot_data(S=0,N=25):
    j=1
    for i in d.non_smooth_index[S:N]:
        plt.subplot(int(np.sqrt(N-S)),int(np.sqrt(N-S)),j)
        plt.plot(range(0,np.size(d.x_test[i])),d.x_test[i],'k')#Input
        plt.plot(range(np.size(d.x_test[i]),np.size(d.x_test[i])+np.size(d.y_test[i])),d.y_test[i],'g')#Original
        j=j+1
fz=30
matplotlib.rc('xtick', labelsize=fz) 
matplotlib.rc('ytick', labelsize=fz)
font = {'family' : 'latex',
        'weight' : 'bold',
        'size'   : fz}

matplotlib.rc('font', **font)



if input('Do you want to generate data in the form required and recall the deeplearned model ?'):
    d = deep(metrics='sl',slid_win=200,thr=2.54210526,sin_pt=0)
    #d.data_import()#import data
    #d.data_shaper()#shape data
    #d.model_recall()#recall model
    ##d.model_save()
    #d.model_main()#generate predictions on the data and generate metric
    d.main_recall()

#Types of training data
    
#Finding the right thresholds--------------------------------------------------
if input('Create the norm list for the y_train? This will generate n_list(all y_train),osc_list,frz_list'):
    n_list=[]
    osc_list=[]
    frz_list=[]
    inter_list=[]
    n_list_train_predictions=[]
    y_train_pred=d.model_sl.predict(d.x_train)
    for i in range(np.shape(d.y_train)[0]):
        n_list.append(-1*d.freeze_liklihood_norm(d.y_train[i].T))#True y_train norms
        n_list_train_predictions.append(-1*d.freeze_liklihood_norm(y_train_pred[i].T))#Norm of y_train predictions
        if -1*d.freeze_liklihood_norm(d.y_train[i].T)>3.2:
            osc_list.append(d.y_train[i].T)
        elif -1*d.freeze_liklihood_norm(d.y_train[i].T)<.1:
            frz_list.append(d.y_train[i].T)
        elif -1*d.freeze_liklihood_norm(d.y_train[i].T)>.1 and -1*d.freeze_liklihood_norm(d.y_train[i].T)<3.2:
            inter_list.append(d.y_train[i].T)
    

if input('Find the best set of thresholds for the classifier derived from the Deep Learning algorithm and a measure?'):
    Kc_star_list_DL=[]
    for Kt in np.linspace(0.1,4,100):
        error_list =[]
        for Kc in np.linspace(0.1,4,100):
            clssifier_predictions = np.array(n_list_train_predictions)<Kc
            true_labels = np.array(n_list)<Kt
            error = np.linalg.norm(true_labels.astype(float)-clssifier_predictions.astype(float))
            error_list.append(error)
        Kc_star_list_DL.append(np.linspace(0.1,4,100)[np.argmin(error_list)])
    
    
    
    plt.plot(np.linspace(0.1,4,100),Kc_star_list_DL,'o',label=r'DRNN')
    plt.title(r'Best Thresholds')
    plt.xlabel(r'$K_{t}$')
    plt.ylabel(r'$K_{c}^{*}$')
    plt.legend()
    plt.tight_layout()
    
    
    
if input('OMP based reconstruction(will not affect resto of the code)?'):
    import pandas as pd
    weights_list=[]
    for weights in d.model_sl.layers[1].get_weights():  
        weights_list.append(weights)

    X  = np.array(weights_list)[0].T
    [dum,dum1,dum2] = extract_cfs(d.y_train,3)
    cf_hist = [sum(dum.ravel()==i) for i in range(0,20)]
    bases_best = sorted(range(len(cf_hist)),reverse=1,key=lambda k: cf_hist[k])[0:5]
    
    [plt.plot(X[:,bases_best[i]],label='Dict-el '+str(bases_best[i]),linewidth=6) for i in range(5)]
    plt.tight_layout()
    plt.title(r'Dictionary Elements')
    plt.xlabel(r'Time Scaled')
    plt.ylabel(r'Amplitude')
    plt.legend()
    plt.tight_layout()
    
    error_wrt_cfno_list = []
    for i in range(1,20):
        [omp_list,y_list] = sparse_recon(d.y_train,i)
        #plt.plot(np.dot(X,omp_list[100]),'r')#Reconstruction
        #plt.plot(y_list[100],'g')#Original
        
        error_recon = [((np.linalg.norm(sc.signal.detrend(np.dot(X,omp_list[i]))-y_list[i],1))/np.linalg.norm(y_list[i],1)) for i in range(len(y_list))]
        #plt.hist(error_recon,200)
        error_wrt_cfno_list.append(np.mean(error_recon))
    plt.plot(error_wrt_cfno_list,linewidth=6)
    plt.title(r'Error in Reconstruction vs OMP coefficients')
    plt.xlabel(r'Number of coefficients used')
    plt.ylabel(r'Error in Reconstruction')
    plt.legend()
    #plt.tight_layout()
    
    
    #Reconstruction plot
    [omp_list,y_list] = sparse_recon(d.y_train,5)
    for i in range(100,110):
    
        plt.plot(y_list[i])
        plt.plot(sc.signal.detrend(np.dot(X,omp_list[i])))
    
    #plot the 5 best basis from the sorted list, and reconstructions
    #compute the reconstruction accuracy over different samples and plot


if input('threshold using Kmeansand update the thershold ?'):  
    from sklearn.cluster import KMeans
    thresholds = []
    for i in range(100):
        kmeans = KMeans(3, random_state=i)
        labels = kmeans.fit(np.array(n_list).reshape(-1,1)).predict(np.array(n_list).reshape(-1,1))
        thresholds.append(np.max(kmeans.cluster_centers_))
    threshold = np.mean(thresholds)
    

if input('threshold using GMM and update the threshold as the middle class one ?'):
    from sklearn.mixture import GaussianMixture as gm
    #from sklearn.mixture import BayesianGaussianMixture as gm
    gm_o = gm(n_components=3)
    gm_o.fit(np.array(n_list).reshape(-1,1))
    threshold= (max(gm_o.means_)[0]-np.sqrt(gm_o.covariances_[np.argmax(gm_o.means_)])*2)[0][0]
    bound = 0.5
    lb=bound-0.01
    ub=bound+0.01
    threshold = np.linspace(0,5,1000)[np.min([np.argmin(np.abs(gm_o.predict_proba(np.linspace(0,5,1000).reshape(-1,1))[:,0]-cutoff)) for cutoff in np.linspace(lb,ub,50)])]#50% probability of the second class'check the value of 'gm_o.predict_proba(np.linspace(0,5,1000).reshape(-1,1))[:,0]' this zero here. may need to change according to the output

    if threshold>max(gm_o.means_):
        print('ERROR')
        
    print('Choosing threshold as the middle class ')
    threshold = [gm_o.means_[i] for i in range(3) if gm_o.means_[i]<3 and gm_o.means_[i]>.5][0][0]
    print(threshold)
    
    
    plt.hist(n_list,1000,normed=1)
    plt.plot(np.linspace(0,5,100),gm_o.predict_proba(np.linspace(0,5,100).reshape(-1,1))[:,1],'g',label=r'Probability to Fall in stepping cluster')
    plt.plot(np.linspace(0,5,100),gm_o.predict_proba(np.linspace(0,5,100).reshape(-1,1))[:,2],'r',label=r'Probability to Fall in freezing cluster')
    plt.title(r'Data Clusters')
    plt.xlabel(r'$\mathrm{l_{2}}$Norm of the data($\mathrm{N_{2}}$)')
    plt.ylabel(r'Probability($\mathrm{P}$)')
    plt.legend()
    #plt.tight_layout()
    

if input('Regenerate metrics of the deep learning network ?'):
    d = deep(metrics='sl',slid_win=200,thr=threshold,sin_pt=0)
    d.main_recall()



if input('Generate correct training and testing data ?'):

    [input_train_r,output_train_r,inp_norm_train_r] = training_data(d)
    [input_train_c,output_train_c,inp_norm_train_c] = training_data_classifier(d,threshold)
    [input_train_f,output_train_f,inp_norm_train_f] = training_data_all(d)
    [input_train_feat,output_train_feat,inp_norm_train_feat] = training_data_classifier_with_feature_extraction(d,threshold)
    
    
    
    [input_test_r,output_test_r,inp_norm_test_r] = testing_data(d)
    [input_test_c,output_test_c,inp_norm_test_c] = testing_data_classifier(d,threshold)
    [input_test_f,output_test_f,inp_norm_test_f] = testing_data_all(d)
    
    if input('Generate input and output test according tho the DRNN for GB(Not-Detrending to make sense of AUROC-GB predictions)?'):
        input_test_f = d.x_test.reshape(d.x_test.shape[0:2])
        output_test_f = d.y_test.reshape(d.y_test.shape[0:2])
    
    [input_test_feat,output_test_feat,inp_norm_test_feat] = testing_data_classifier_with_feature_extraction(d,threshold)
    
    clf_r = rf_r(max_depth=2000, random_state=15)
    clf_c = rf_c(max_depth=2000, random_state=15)
    #clf_f  = multiout_r(GradientBoostingRegressor(random_state=0),n_jobs=-1)#Defined but not fit
    clf_c_features =rf_c(max_depth=2000, random_state=15)
    
    
    
    clf_r.fit(input_train_r,output_train_r)
    clf_c.fit(input_train_c,output_train_c)
    clf_f = joblib.load('clf_f_GB_matching_DRNN')
    
    clf_c_features.fit(np.array(input_train_feat),output_train_feat)
    
    print(clf_c.score(input_test_c,output_test_c))
    print(clf_r.score(input_test_r,output_test_r))
    print(clf_c_features.score(np.array(input_test_feat),output_test_feat))

if input('Find the best set of thresholds for the classifier derived from the GB algorithm and a measure?'):
    
    clf_f = joblib.load('clf_f_GB_matching_DRNN')
    GB_y_train_pred = clf_f.predict(input_train_f)
    n_list_y_train_pred_Gradboost = [-d.freeze_liklihood_norm(y_test=GB_y_train_pred[i]) for i in range(0,GB_y_train_pred.shape[0])]

    Kc_star_list_GB=[]
    for Kt in np.linspace(0.1,4,100):
        error_list =[]
        for Kc in np.linspace(0.1,4,100):
            clssifier_predictions = np.array(n_list_y_train_pred_Gradboost)<Kc
            true_labels = np.array(n_list)<Kt
            error = np.linalg.norm(true_labels.astype(float)-clssifier_predictions.astype(float))
            error_list.append(error)
        Kc_star_list_GB.append(np.linspace(0.1,4,100)[np.argmin(error_list)])
    
    
    
    plt.plot(np.linspace(0.1,4,100),Kc_star_list_GB,'o',label=r'GB')
    plt.title(r'Best Thresholds')
    plt.xlabel(r'$K_{t}^{*}$')
    plt.ylabel(r'$K_{c}^{*}$')
    plt.legend()
    plt.tight_layout()


if input    ('fit a GB without detrending?'):
    
    clf_f  = multiout_r(GradientBoostingRegressor(random_state=0))#Defined but not fit
    #clf_f.fit(d.x_train.reshape(d.x_train.shape[0:2]),d.y_train)
    clf_f = joblib.load('clf_f_GB_matching_DRNN')
    #joblib.dump(clf_f,'clf_f_GB_matching_DRNN')




if input('Generate a good basis for the data using k means?'):
    from sklearn.cluster import KMeans
    kmeans = KMeans(4, random_state=i)
    kmeans.fit(d.y_train.reshape(d.y_train.shape[0:2]))
    plt.plot(kmeans.cluster_centers_.T)
    plt.scatter(kmeans.labels_,output_train_r)

if False:
    plt.scatter(output_test,clf.predict(input_test))
    plt.scatter(output_train,clf.predict(input_train))
    
    plt.scatter(inp_norm_train,output_train)
    plt.scatter(inp_norm_test,output_test-clf.predict(input_test))


# For classifier
if input('Generate ROC Curves for deep learning and RF classifiers?'):
    score_list =[]
    fpr_list, tpr_list, thr_holds_list = ([],[],[])
    for thr in np.linspace(0.1,4,100):
        [input_train,output_train,inp_norm_train] = training_data_classifier(d,thr)
        [input_test,output_test,inp_norm_test] = testing_data_classifier(d,thr)
        clf_c.fit(input_train,output_train)
        score_list.append(clf_c.score(input_test,output_test))
        fpr_list, tpr_list, thr_holds_list= metr.roc_curve(output_test,clf_c.predict_proba(input_test)[:,1],pos_label=True)
    
    fpr, tpr, thr_holds= metr.roc_curve(output_test,clf_c.predict_proba(input_test)[:,1],pos_label=True)
    #ROC curve for the deep learned classifier
    
    for d.thr in [2,3,3.7,3.9]:
        d.metrics_gen_sl()
        plt.plot(d.fpr, d.tpr,linewidth=6,label=r'$K_{c}^{*}$='+str(d.thr))
        #plt.plot(fpr, tpr, label='ROC-Random Forest')
        plt.plot(d.fpr, d.fpr,'k')
    plt.xlabel(r'False Positive Rate')
    plt.ylabel(r'True Positive Rate')
    plt.title(r'ROC Curve')
    plt.tight_layout()
    plt.legend()
        
if input('Generate norm distance for testing the accuracy of the time series prediction?'):
    y_sklearn_pred = clf_f.predict(input_test_f)
    if input('distance to be measured using correlation distance?'):
        dists_sklearn = np.diag(sc.spatial.distance.cdist(clf_f.predict(input_test_f),output_test_f,'correlation'))
        dists_deep_rnn = np.diag(sc.spatial.distance.cdist(d.y_pred_sl,output_test_f,'correlation'))
    
    dists_sklearn = np.array([np.linalg.norm((sc.signal.detrend(y_sklearn_pred)-sc.signal.detrend(output_test_f))[i]/(np.linalg.norm(output_test_f[i]))) for i in range(2496)])
    dists_deep_rnn = np.array([np.linalg.norm((sc.signal.detrend(d.y_pred_sl)-sc.signal.detrend(output_test_f))[i])/(np.linalg.norm(output_test_f[i])) for i in range(2496)])

    plt.hist(dists_deep_rnn,1000,normed=True)
    plt.hist(dists_sklearn,1000,normed=True)
    
    if input('Animation of the time series prediction?'):
        for i in range(1000):
            plt.plot(sc.signal.detrend(d.y_pred_sl[i]),label='RNN')
            plt.plot(sc.signal.detrend(output_test_f)[i],label='Original')
            plt.plot(y_sklearn_pred[i],label='GradientBoost')
            plt.ylim(-0.8,0.8)
            plt.legend()
            plt.pause(0.1)
            plt.clf()


if input('Computation for plotting the same norm graphs for RNN ?'):
    #Computation for plotting the same norm graphs for RNN
    n_list_y_test_pred = [-d.freeze_liklihood_norm(y_test=d.y_pred_sl[i]) for i in range(0,2496)]
    n_list_y_test = [-d.freeze_liklihood_norm(y_test=d.y_test[i]) for i in range(0,2496)]
    
    #RF regression
    n_list_y_test_pred_r = clf_r.predict(input_test_r)
    
    #Gradient BOOSTing
    n_list_y_test_pred_Gradboost = [-d.freeze_liklihood_norm(y_test=y_sklearn_pred[i]) for i in range(0,2496)]
    
    if False:
        sc.io.savemat('scatter3d_norepred_rnn2',{'x':n_list_x_test,'y':n_list_y_test ,'z':n_list_y_test_pred,'x_t':np.array(n_list_x_test)<3.45,'y_t':np.array(n_list_y_test)<3.45 ,'z_t':np.array(n_list_y_test_pred)<3.45})#was 2.5
        sc.io.savemat('scatter3d_norepred_rf_regression2',{'x_r':inp_norm_test_r,'y_r':output_test_r ,'z_r':n_list_y_test_pred_r})
        sc.io.savemat('scatter3d_norepred_rf_classifier2',{'x_c':inp_norm_test_c,'y_c':output_test_c ,'y_r':output_test_r ,'z_c':clf_c.predict(input_test_c)})
        sc.io.savemat('scatter3d_norepred_GB',{'x':inp_norm_test_r,'y':n_list_y_test,'z':n_list_y_test_pred_Gradboost})
    #plt.scatter(np.abs(d.y_likelihood_true),np.abs(d.y_likelihood))
    precision = (d.tp+0.001)/(d.tp+d.fp)
    recall = (d.tp+0.001)/(d.tp+d.fn)
    F1 = (2*precision*recall)/(precision+recall)
    
    
if input('Analyze the deeplearning imaginary classifiers for different threshold values DRNN?'):
    auc_rnn_list = []
    for threshold in np.linspace(0.1,4,300):
        d.thr = threshold
        d.metrics_gen_sl()
        auc_rnn_list.append(d.auc_score)
        print(d.auc_score)

if input('Analyze the GB imaginary classifiers for different threshold values GB?'):
    auc_GB_list = []
    y_sklearn_pred = clf_f.predict(input_test_f)#For GB
    for threshold in np.linspace(0.1,4,300):
        d.thr = threshold
        y_true_GB = []
        y_likelihood_GB =[]
        true_sums= []
        for i in range(np.shape(output_test_f)[0]):
            y_true_GB.append(d.freeze_detect_norm(output_test_f[i],index=i)=='Freezing')
            y_likelihood_GB.append(d.freeze_liklihood_norm(y_sklearn_pred[i]))
        auc_score_GB = metr.roc_auc_score(y_true_GB,y_likelihood_GB)
        print(auc_score_GB)
        auc_GB_list.append(auc_score_GB)
    
if input('Analyze the classifier for the AUROC plot'):
    auc_score_RFC_list=[]
    auc_score_RFCF_list=[]
    true_sums_list =[]
    for threshold in np.linspace(0.1,4,300):
        d.thr = threshold
        [input_test_c,output_test_c,inp_norm_test_c] = testing_data_classifier(d,threshold)
        true_sums_list.append(np.sum(np.array(output_test_c)==True))
        print(true_sums_list)
        #[input_train_r,output_train_r,inp_norm_train_r] = training_data(d)
        [input_train_c,output_train_c,inp_norm_train_c] = training_data_classifier(d,threshold)

        [input_train_feat,output_train_feat,inp_norm_train_feat] = training_data_classifier_with_feature_extraction(d,threshold)

        [input_test_feat,output_test_feat,inp_norm_test_feat] = testing_data_classifier_with_feature_extraction(d,threshold)
        clf_c = rf_c(max_depth=2000, random_state=15)
        clf_c_features =rf_c(max_depth=2000, random_state=15)
        clf_c.fit(input_train_c,output_train_c)
        clf_c_features.fit(np.array(input_train_feat),output_train_feat)
        print((metr.roc_auc_score(output_test_c,clf_c.predict_proba(input_test_c)[:,1]),metr.roc_auc_score(output_test_feat,clf_c_features.predict_proba(input_test_feat)[:,1])))
        auc_score_RFC_list.append(metr.roc_auc_score(output_test_c,clf_c.predict_proba(input_test_c)[:,1]))
        auc_score_RFCF_list.append(metr.roc_auc_score(output_test_feat,clf_c_features.predict_proba(input_test_feat)[:,1]))
        
if input('Generate other pretty plots for the TBME paper?'):
    plt.figure()
    plt.plot(np.linspace(0.1,4,300),auc_GB_list,'r',label=r'GB',linewidth=6)
    plt.plot(np.linspace(0.1,4,300),auc_rnn_list,'g',label = r'DRNN',linewidth=6)
    plt.plot(np.linspace(0.1,4,300),auc_score_RFC_list,'k',label = r'RFC',linewidth=6)
    plt.plot(np.linspace(0.1,4,300),auc_score_RFCF_list,'y',label = r'RFCF',linewidth=6)
    plt.plot(np.linspace(0.1,4,300),1-((np.array(true_sums_list))/(np.size(output_test_c)+0.00)),'b',label = r'BL',linewidth=6)
    
    plt.plot(np.linspace(0.1,4,299),1+10*np.diff(1-((np.array(true_sums_list))/(np.size(output_test_c)+0.00))),'*',label = r'DBL',linewidth=1)
    plt.title(r'AUROC for Different Thresholds')
    plt.xlabel(r'Threshold')
    plt.ylabel(r'AUROC')
    plt.legend()
    plt.tight_layout()
    
if input('Find the best AUROC for the specified threshold?'):
    auc_GB_list[np.argmin(abs(np.linspace(0.1,4,300).astype(float)-threshold))]
    auc_rnn_list[np.argmin(abs(np.linspace(0.1,4,300).astype(float)-threshold))]
    auc_score_RFC_list[np.argmin(abs(np.linspace(0.1,4,300).astype(float)-threshold))]
    auc_score_RFCF_list[np.argmin(abs(np.linspace(0.1,4,300).astype(float)-threshold))]
    1-((np.array(true_sums_list))/(np.size(output_test_c)+0.00))[np.argmin(abs(np.linspace(0.1,4,300).astype(float)-threshold))]#Baseline
#Completely new trick - Improving the accuracy of time series prediction
#By using the better predicted norms
if False:
    from sklearn.mixture import GaussianMixture as gm
    gm_o = gm(n_components=3)
    
    error_rf = np.array([output_train_r,abs(output_train_r-clf_r.predict(input_train_r))]).T
    #error_rf = np.array([(output_train_r-clf_r.predict(input_train_r))]).T
    
    grid = np.array([output_train_r,abs(output_train_r-clf_r.predict(input_train_r))]).T
    #grid = np.array(np.linspace(0,3,100))
    
    gm_o.fit(np.array(error_rf))
    #gm_o.fit(np.array(error_rf).reshape(-1,1))
    
    plt.scatter(grid[:,0],gm_o.predict_proba(grid)[:,0])
    #gm_o.predict_proba(grid.reshape(-1,1))

    # Learning from the error
    from sklearn.linear_model import BayesianRidge as BR
    from sklearn.linear_model import SVM
    
    
    
    y_train_pred= d.model_sl.predict(d.x_train)
    n_list_y_train_pred = np.array([-d.freeze_liklihood_norm(y_test=y_train_pred[i]) for i in range(0,4160)])
    
    output_error_train = (output_train_r)
    input_error_train = np.array([clf_r.predict(input_train_r),n_list_y_train_pred]).T
    
    output_error_test = (output_test_r)
    input_error_test = np.array([clf_r.predict(input_test_r),n_list_y_test_pred]).T
    
    
    clf_error = rf_r(max_depth=10, random_state=1)
    clf_error.fit(input_error_train,output_error_train)
    
    
    clf_error.score(input_error_test,output_error_test)
    
    error_test_pred = clf_error.predict(input_error_test)
    error_train_pred = clf_error.predict(input_error_train)
    
    cum_error_list_test =[np.mean([abs(error_test_pred[i]-output_error_test[i]) for i in range(2496) if output_test_r[i]>th]) for th in np.linspace(0.1,4,100)]

    cum_error_list_train =[np.mean([abs(error_train_pred[i]-output_error_train[i]) for i in range(4160) if n_list_y_train_pred[i]>th]) for th in np.linspace(0,5,1000)]


    #Algorithm starts here
    
if False:
    d.sliding_window=50
    d.window_len=150
    d.data_shaper()
    



if False:#Feature based learning
    #[dum,x_train_cf,dum] = extract_cfs(d.x_train[:,:,0],1)#The complicated extract cf function is still retained to 
    [dum,y_train_cf,dum] = extract_cfs(d.y_train,1)
    t1 = np.nonzero(y_train_cf)[1]
    t2 = y_train_cf[np.nonzero(y_train_cf)]
    y_train_features = [t1,t2]
    #[dum,x_test_cf,dum] = extract_cfs(d.x_test[:,:,0],1)
    [dum,y_test_cf,dum] = extract_cfs(d.y_test,1)
    t1 = np.nonzero(y_test_cf)[1]
    t2 = y_test_cf[np.nonzero(y_test_cf)]
    y_test_features = [t1,t2]
    
    output_train_basis = np.array(y_train_features).T
    output_test_basis = np.array(y_test_features).T

    if False:
        [input_test_f,dum,dum] = testing_data_all(d)
        [input_train_f,dum,dum] = training_data_all(d)
    if True:
        [dum,x_train_cf,dum] = extract_cfs(d.x_train[:,:,0],5)
        t1 = np.nonzero(x_train_cf)[1]
        t2 = y_train_cf[np.nonzero(x_train_cf)]
        x_train_features = [t1,t2]
        [dum1,x_test_cf,dum] = extract_cfs(d.x_test[:,:,0],5)
        t1 = np.nonzero(x_test_cf)[1]
        t2 = y_test_cf[np.nonzero(x_test_cf)]
        x_test_features = [t1,t2]
    

if False:
    #Trainin the regressor for time series prediction with the new extracted features
    clf_r_basis1 = rf_r(max_depth=2000, random_state=15)
    clf_c_basis1 = rf_c(max_depth=2000, random_state=2)
    clf_c_basis1 = SVC()
    
    clf_c_basis1.fit(input_train_f,output_train_basis[:,0])
    clf_c_basis1.score(input_test_f,output_test_basis[:,0])
    

import pywt
if False:
    
    for i in range(1000):
        plt.subplot(2,1,1)
        plt.plot(d.x_train[i,:][:,0])
        coef, freqs=pywt.cwt(sc.signal.detrend(d.x_train[i,:][:,0]),3,'mexh')
        plt.subplot(2,1,2)
        plt.plot(coef.T[0:10])
        plt.ylim((0,2))
        plt.pause(.1)
        plt.clf()

if False:#Feature reduction and further extrction
    from matplotlib import mlab as ml
    for i in range(1000):
        
        #Just plotting
        plt.subplot(2,1,1)
        plt.plot(d.x_train[i,:][:,0])
        z = ml.specgram(d.x_train[i,:][:,0],NFFT=100,noverlap=0)[0]
        plt.subplot(2,1,2)
        #plt.contour(z)
        #plt.plot(z.T)
        plt.plot(z.T[:,0:1]-np.mean(z.T[:,0:1]),'r')
        plt.ylim((0,10))
        #plt.matshow(z.T)
        plt.pause(0.1)
        plt.clf()
    z_list = []  
    for i in range(max(d.y_train.shape)):
        z = ml.specgram(d.x_train[i,:][:,0],NFFT=100,noverlap=0)[0]
        z_list.append(z.T[:,0:1])#-np.mean(z.T[:,0:1]))
        

        