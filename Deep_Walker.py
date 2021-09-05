from __future__ import print_function,division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM ,Conv1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import mlab as ml
import keras
import matplotlib 
import scipy as sc
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)
class deep_learner:
    def __init__(self,metrics='sl',slid_win=200,thr=0.4,sin_pt=0,epch=1):
        '''If the window len is changed and the shaper is called x_train, x_test , y_train, y_test shapes wil change and there will be no approprate change 
        #in the predicted data and therefore, metrics generated'''
        
        #Subject size
        self.single_pat=sin_pt
        self.sliding=True
        self.metrics_gen_alg = metrics
        #Hyper parameters
        self.epoch_sl = epch
        self.epoch_sf = epch
        self.scaler=MinMaxScaler()
        self.thr = thr
        #Data Import
        self.data_import()
        self.data_size = np.size(self.data_input)
        self.window_len =400#Should divide data size training is done for this size if changed can't call recall or call appropriate recall

        self.sliding_window=slid_win#Should divide window len
        self.predict_len = 150#Should be less than the window len
        self.samples = int(self.data_size/self.window_len)
        self.sample_train_size = int(self.samples*63/100)
        self.sample_test_size = self.samples-self.sample_train_size
        self.pat_perc=70/100.0
        
        self.unseen_test=False
        
        npat_train=int(self.data_list.shape[0]*self.pat_perc)
        self.data_list = self.data_list[0:npat_train]#Data list changes to new prtitioned set
        
        
        if self.unseen_test==True:
            self.data_list = self.data_list[npat_train:-1]
        print('learner_initiated with data')
        
    def model_recall(self):

        self.model_sf = keras.models.load_model('../model_sf_g_3000.h5')
        self.model_sl = keras.models.load_model('../model_sl_g_3000.h5')
        print('recalling the model')
        
    def model_gen(self):
        '''uses the current x_train and y_train to generate a neural model'''
        print('generating the model')
        #stateless
        model_sl = Sequential()
        model_sl.add(LSTM(20,
                  input_shape=(self.window_len,1),
                  batch_size=self.sample_train_size))
        model_sl.add(Dense(self.predict_len))
        model_sl.compile(loss='mse', optimizer='adam')
        model_sl.fit(x=self.x_train,y=self.y_train,epochs=self.epoch_sl,validation_split=0.2)
        #Statefull
        bat_sf = 1
        model_sf = Sequential()
        model_sf.add(LSTM(20,
                  input_shape=(self.window_len,1),
                  batch_size=bat_sf,stateful='True'))
        model_sf.add(Dense(self.predict_len))
        model_sf.compile(loss='mse', optimizer='adam')
        model_sf.fit(x=self.x_train,y=self.y_train,epochs=self.epoch_sf,batch_size=bat_sf,validation_split=0.2)
        self.model_sf = model_sf
        self.model_sl = model_sl
        
    def model_save(self):
        import time
        self.model_sf.save('model_sf_g'+str(time.localtime()[0:6])+'.h5')
        self.model_sl.save('model_sl_g'+str(time.localtime()[0:6])+'.h5')
    def data_import(self):
        
        if self.single_pat==True:
            #d1 = np.genfromtxt('../../Data/FD27_1.txt')
            #d2 = np.genfromtxt('../../Data/FD42_2.txt')
            d3 = np.genfromtxt('../../../Data/FD78_1.txt')
            self.data_input = d3[4:-1][:,9][0:8000]
        else:
            self.data_list = np.load('../data_list.npy')#Data in numpy 
            self.data_input = self.data_list[0]
        

    def chopper(self,data_input):
        ''' input:[8000] output [x,150], ready for training which may be further 'staked' before training in a large chunk'''
        
        
        data_size = np.size(data_input)
        window_len =400#Should devide data size
        samples = int(data_size/window_len)
        sample_train_size = int(samples*63/100)
        sample_test_size = samples-sample_train_size
        
        
        data_input = self.scaler.fit_transform(np.array(data_input).reshape(-1,1))
        data_split=np.reshape(data_input,(samples,window_len))
        if self.sliding==True:#The sliding window creates monstrous amount of data
            data_split = self.slider(data_input)[::self.sliding_window]
            samples = np.shape(data_split)[0]
            sample_train_size = int(samples*63/100)
            sample_test_size = samples-sample_train_size
        
        x_train = data_split[0:sample_train_size]
        x_test = data_split[sample_train_size:]
        y_train = np.zeros((sample_train_size,self.predict_len))
        y_test = np.zeros((sample_test_size,self.predict_len))
        
        
        
        
        step = int(self.window_len/self.sliding_window)
        if step==0:
            print('error')
            import winsound
            winsound.Beep(250,1000)
            
        for i in range(np.shape(x_train)[0]-(1+step)):
            y_train[i,:]  = x_train[i+step,:][0:self.predict_len]
        
        for i in range(np.shape(x_test)[0]-(1+step)):
            y_test[i,:] = x_test[i+step,:][0:self.predict_len]    
        self.step=step
        
        # reshape input to be [samples, time steps, features]
        x_train = np.reshape(x_train,(sample_train_size,self.window_len,1))
        x_test = np.reshape(x_test,(sample_test_size,self.window_len,1))
        y_train = np.reshape(y_train,(sample_train_size,self.predict_len))
        return [x_train[0:-(self.step+1)],y_train[0:-(self.step+1)],x_test[0:-(self.step+1)],y_test[0:-(self.step+1)]]
    
    def slider(self,data_input):
        
        '''input: 8000 dim vector gives out split data after sliding for a particular window length output:[x,150]'''
        data_split = []
        data_size = np.size(data_input)
        
        for i in range(data_size-self.window_len):
            data_split.append(data_input[i:self.window_len+i])
        data_split = np.array(data_split)
        samples = np.shape(data_split)[0]
        data_split =np.reshape(data_split,(samples,self.window_len))
        return data_split

    def data_shaper(self):
        '''Returns parted slided xtrain xtest and ytrain ytest'''
        
        if True:
            self.data_split = self.slider(self.data_input)[::self.sliding_window]
            self.samples = np.shape(self.data_split)[0]
            self.sample_train_size = int(self.samples*63/100)
            self.sample_test_size = self.samples-self.sample_train_size
        
        

            
        
        [self.x_train,self.y_train,self.x_test,self.y_test]=self.chopper(self.data_list[0])
        ind=0
        for data_input in self.data_list:
            if np.shape(data_input)[0]==8000:
                #print(np.shape(data_input)[0]==8000)
                ind=ind+1
                [x_train_n,y_train_n,x_test_n,y_test_n]=self.chopper(data_input)
                self.x_train = np.vstack((self.x_train,x_train_n))
                self.x_test = np.vstack((self.x_test,x_test_n))
                self.y_train = np.vstack((self.y_train,y_train_n))
                self.y_test = np.vstack((self.y_test,y_test_n))
            else :
                pass
    
    def freeze_detect_fourier(self,y_test):
        y_test_dt = sc.signal.detrend(y_test)
        y_test_fr = np.abs(np.fft.fft(y_test_dt))
        if max(y_test_fr)<4:
            out = 'Freezing'
        else:
            out  = 'Not Freezing'#freezing implies True
        return out

    def freeze_detect_norm(self,y_test,index='not given'):
        self.y_test_dt = sc.signal.detrend(y_test)[-1*self.predict_len:]
        self.y_test_norm = np.abs(np.linalg.norm(self.y_test_dt))
        
        if self.y_test_norm<0.01:
            print('Error')
            print(index)
            #plt.plot(self.y_test_dt)
        
        
        
        #print(y_test_norm)
        if self.y_test_norm<self.thr:
            out = 'Freezing'
        else:
            out  = 'Not Freezing'#freezing implies True
        return out
        
    
    def freeze_liklihood_norm(self,y_test):
        y_test_dt = sc.signal.detrend(y_test)[-1*self.predict_len:]
        y_test_norm = np.abs(np.linalg.norm(y_test_dt))
        #print(y_test_norm)
        out = -y_test_norm
    
        return out
    def predictions_gen(self):
        self.y_pred_sl = self.model_sl.predict(self.x_test)
        self.y_pred_sf = self.model_sf.predict(self.x_test,batch_size=1)
        print('predictions')
        
    def metrics_gen_sl(self):
        
        
        
        from sklearn.metrics import confusion_matrix
        import sklearn.metrics as metr
        self.detection_acc=[]
        self.y_true =[]
        self.y_pred = []
        self.y_likelihood = []
        self.y_likelihood_true=[]
        self.x_test_input=[]
        for i in range(np.shape(self.y_test)[0]):
            self.detection_acc.append(self.freeze_detect_norm(self.y_test[i],index=i)==self.freeze_detect_norm(self.y_pred_sl[i],index=i))
            self.x_test_input.append(self.freeze_detect_norm(np.reshape(self.x_test,(self.x_test.shape[0],self.x_test.shape[1]))[i],index=i)=='Freezing')
            
            self.y_true.append(self.freeze_detect_norm(self.y_test[i],index=i)=='Freezing')
            self.y_pred.append(self.freeze_detect_norm(self.y_pred_sl[i],index=i)=='Freezing')
            self.y_likelihood.append(self.freeze_liklihood_norm(self.y_pred_sl[i]))
            self.y_likelihood_true.append(self.freeze_liklihood_norm(self.y_test[i]))
            
        self.non_smooth_acc=[]
        self.non_smooth_index =[]
        for i in range(np.shape(self.x_test_input)[0]):
            if self.x_test_input[i]!=self.y_true[i]:
                self.non_smooth_acc.append(((self.x_test_input[i]==self.y_pred[i])))
                self.non_smooth_index.append(i)
    
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        self.specificity = (self.tn+0.0000001 )/ (self.tn+self.fp)
        self.sensitivity = (self.tp+0.0000001)/(self.tp+self.fn)
        self.accuracy = (self.tp+self.tn+0.0000001)/(self.tp+self.tn+self.fp+self.fn)
        self.fpr, self.tpr,self.thr_holds= metr.roc_curve(self.y_true,self.y_likelihood,pos_label=True)
        self.auc_score = metr.roc_auc_score(self.y_true,self.y_likelihood)
        
        self.freeze_detect_measure = (1-float(sum(self.non_smooth_acc))/np.size(self.non_smooth_acc))
        print('accuracy,specifiicty,sensitivity')
        print(self.accuracy,self.specificity,self.sensitivity,self.freeze_detect_measure)
        
        
    def metrics_gen_sf(self):
        from sklearn.metrics import confusion_matrix
        import sklearn.metrics as metr
        self.detection_acc=[]
        self.y_true =[]
        self.y_pred = []
        self.y_likelihood = []
        self.y_likelihood_true=[]
        self.x_test_input=[]
        for i in range(np.shape(self.y_test)[0]):
            self.detection_acc.append(self.freeze_detect_norm(self.y_test[i],index=i)==self.freeze_detect_norm(self.y_pred_sf[i],index=i))
            self.x_test_input.append(self.freeze_detect_norm(np.reshape(self.x_test,(self.x_test.shape[0],self.x_test.shape[1]))[i],index=i)=='Freezing')
            
            self.y_true.append(self.freeze_detect_norm(self.y_test[i],index=i)=='Freezing')
            self.y_pred.append(self.freeze_detect_norm(self.y_pred_sf[i],index=i)=='Freezing')
            self.y_likelihood.append(self.freeze_liklihood_norm(self.y_pred_sf[i]))
            self.y_likelihood_true.append(self.freeze_liklihood_norm(self.y_test[i]))
            
        self.non_smooth_acc=[]
        self.non_smooth_index =[]
        for i in range(np.shape(self.x_test_input)[0]):
            if self.x_test_input[i]!=self.y_true[i]:
                self.non_smooth_acc.append(((self.x_test_input[i]==self.y_pred[i])))
                self.non_smooth_index.append(i)
    
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        self.specificity = (self.tn+0.0000001 )/ (self.tn+self.fp)
        self.sensitivity = (self.tp+0.0000001)/(self.tp+self.fn)
        self.accuracy = (self.tp+self.tn+0.0000001)/(self.tp+self.tn+self.fp+self.fn)
        self.fpr, self.tpr,self.thr_holds= metr.roc_curve(self.y_true,self.y_likelihood,pos_label=True)
        self.freeze_detect_measure = (1-float(sum(self.non_smooth_acc))/np.size(self.non_smooth_acc))
        
        
        print('accuracy,specifiicty,sensitivity')
        print(self.accuracy,self.specificity,self.sensitivity)
        
    def model_main(self):
        print('computing the metrics')
        self.predictions_gen()
        if self.metrics_gen_alg=='sl':
            self.metrics_gen_sl()
        else:
            self.metrics_gen_sf()
        
    def main_recall(self):
        self.data_import()#import data
        self.data_shaper()#shape data
        self.model_recall()#recall model
        print('computing the metrics')
        self.predictions_gen()
        if self.metrics_gen_alg=='sl':
            self.metrics_gen_sl()
        else:
            self.metrics_gen_sf()
        
    def main_train(self):
        self.data_import()#import data
        self.data_shaper()#shape data
        self.model_gen#recall model
        print('computing the metrics')
        self.predictions_gen()
        self.model_save()
        if self.metrics_gen_alg=='sl':
            self.metrics_gen_sl()
        else:
            self.metrics_gen_sf()

    def plot_metrics(self):
        print('plotting.....')
        
class deep_learner_feature_based(deep_learner):
    def __init__(self,metrics='sl',slid_win=200,thr=0.4,win_len=150,pr_len=150):
        #Subject size
        self.single_pat=0
        self.sliding=True
        self.metrics_gen_alg = metrics
        #Hyper parameters
        self.epoch_sl = 1
        self.epoch_sf = 1
        self.scaler=MinMaxScaler()
        self.thr = thr
        #Data Import
        self.data_import()
        self.data_size = np.size(self.data_input)
        self.window_len =win_len#Should devide data size
        self.predict_len = pr_len#Should be less than the window len
        self.pat_perc=70/100.0
        self.sliding_window=slid_win
        print('learner_initiated with data')
        self.model_recall()
        
    def feature_extractor_wt(self,y):
        
        
        #self.model_main()
        weights_list=[]
        for weights in self.model_sl.layers[1].get_weights():  
            weights_list.append(weights)
            
        self.X = np.array(weights_list)[0].T
        
        self.omp = OrthogonalMatchingPursuit(n_nonzero_coefs=2)
        self.omp.fit(self.X, y.T)
        
        cf = self.omp.coef_
        ind = np.array(np.nonzero(self.omp.coef_),dtype=int)
        return [cf,ind]
            
    def extract_main(self):
        self.data_shaper()
        
        self.x_train_mat = np.reshape(self.x_train,(self.x_train.shape[0],self.x_train.shape[1]))
        self.x_test_mat = np.reshape(self.x_test,(self.x_test.shape[0],self.x_test.shape[1]))
        
        self.x_train_features=[]
        self.y_train_features=[]
        self.x_test_features=[]
        self.y_test_features=[]
        for y in self.x_train_mat:
            self.x_train_features.append(self.feature_extractor_wt(y)[1])
            
        for y in self.y_train:
            self.y_train_features.append(self.feature_extractor_wt(y)[1])
            
        for y in self.x_test_mat:
            self.x_test_features.append(self.feature_extractor_wt(y)[1])
            
        for y in self.y_test:
            self.y_test_features.append(self.feature_extractor_wt(y)[1])
            
        self.x_train_features = np.array(self.x_train_features)
        self.x_test_features = np.array(self.x_test_features)
        self.y_train_features = np.array(self.y_train_features)
        self.y_test_features = np.array(self.y_test_features)
        
    def plot_data(self,S=0,N=25):
        j=1
        for i in range(S,N):
            plt.subplot(int(np.sqrt(N-S)),int(np.sqrt(N-S)),j)
            plt.plot(range(0,np.size(self.x_test[i])),self.x_test[i],'k')#Input
            plt.plot(range(np.size(self.x_test[i]),np.size(self.x_test[i])+np.size(self.y_test[i])),self.y_test[i],'g')#Original
            j=j+1
    
    def nondeep_learner(self,X,y):
        from sklearn.ensemble import RandomForestRegressor as rf
        self.model_f = rf(max_depth=10, random_state=0)
        self.model_f.fit(X, y)
        

        
    def learn_features(self):
        self.fit_data(self.x_train_features,self.y_train_features)
        
        
        
        X = np.reshape(self.x_train_features,(self.x_train_features.shape[0],self.x_train_features.shape[2]))

        
        y = np.reshape(self.y_train_features,(self.y_train_features.shape[0],self.y_train_features.shape[2]))
        
        X_d = np.reshape(self.x_train_features,(self.x_train_features.shape[0],self.x_train_features.shape[2],1))
                
    def deep_learner_rnn(X,y,samples=200,output_shape=1600,inputshape=4,epoch=300):
        #y = np.reshape(X,(samples,output_shape))
        #X = np.reshape(y,(samples,inputshape))
    
        #window_len  = inputshape
        #predict_len=output_shape
        epoch_sf=epoch
        model_sf = Sequential()
        model_sf.add(LSTM(20,
                  input_shape=(inputshape,1),
                  batch_size=samples))
        model_sf.add(Dense(output_shape, input_shape=(inputshape,)))
        model_sf.compile(loss='mse', optimizer='adam')
        model_sf.fit(x=X,y=y,epochs=epoch_sf)
        return model_sf
    
    def deep_learner_conv(X,y,samples=200,output_shape=150,inputshape=400,epoch=300):
        epoch_sf=epoch
        model_sf = Sequential()
        model_sf.add(Conv1D(20,100,input_shape=(None,inputshape)))
        #model_sf.add(Dense(output_shape))
        model_sf.compile(loss='mse', optimizer='adam')
        model_sf.fit(x=X,y=y,epochs=epoch_sf)
        return model_sf

        