
# CODE FOR Python3.11
# Imports
# Basic libraries
#    !pip install pandas numpy seaborn matplotlib scikit-learn scipy statsmodels demoji
#    !pip install -U sentence-transformers
#    !pip install umap-learn
#    !pip install xgboost
#    !pip install "tensorflow<2.16"
#    !pip install tf-keras
#    !pip install --force-reinstall --no-cache-dir "numpy<2" "ml-dtypes<0.3.2"
#    !pip install scikit-optimize
#    !pip install GPyOpt
import pandas as pd
import numpy as np
import sklearn
import random
import pickle as pk
import os
import scipy
import demoji
from sentence_transformers import SentenceTransformer


for embed_model in ['sentence-transformers/distiluse-base-multilingual-cased-v2']:#["Qwen/Qwen3-Embedding-0.6B",'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2','stsb-xlm-r-multilingual','sentence-transformers/distiluse-base-multilingual-cased-v2']:
    print(embed_model)
    embedding_dim = 768
    if embed_model in ['sentence-transformers/distiluse-base-multilingual-cased-v2']:
        embedding_dim = 512
    elif embed_model in ['sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2']:
        embedding_dim = 384
    elif embed_model in ["Qwen/Qwen3-Embedding-0.6B"]:
        embedding_dim = 1024
    model = SentenceTransformer(embed_model)
    from sklearn.model_selection import GridSearchCV
    import xgboost as xgb 
    from xgboost import XGBClassifier
    import tensorflow as tf
    from tensorflow.keras import regularizers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout, LSTM, InputLayer, BatchNormalization, LeakyReLU, Normalization, Softmax
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.optimizers.schedules import ExponentialDecay
    from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
    from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
    from sklearn.metrics import accuracy_score,roc_auc_score,f1_score
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from scipy.stats import spearmanr
    import statsmodels
    import skopt
    from skopt.space import Real, Integer,Categorical
    from skopt import BayesSearchCV
    import GPyOpt
    from GPyOpt.methods import BayesianOptimization
    n=10
    print('embedding')
    batches = [GT_labels['text'].values[i:i+n] for i in range(0, len(GT_labels['text'].values), n)] 
    embeddings = [model.encode(batch) for batch in batches]
    embeddings = [e for ee in embeddings for e in ee]
    GT_labels['embeddings'] = [e for e in embeddings]
    gt_data = GT_labels.sample(frac=1)# mix up the rows
    # LEGACY RESULTS - THESE ARE NOT USED IN THE PAPER
    gpt_file = 'gpt35_qualtrics-prompt_responses.csv'
    gpt_labels = pd.read_csv(gpt_file)
    gpt_texts = set(gpt_labels['text'].values.tolist())
    response_gpt_labels = gpt_labels.groupby('text')
    gpt_pred = []
    for text in gt_data['old_text'].values:
        label = 0
        if text in gpt_texts:
            response = response_gpt_labels.get_group(text)['responses'].values[0]
            if 'Yes' in response[:4]:
                label = 1
    
            elif 'No' in response[:4]:
                label = 0
        else:
          print('error')
        gpt_pred.append(label)
    
    gt_data['gpt_pred'] = gpt_pred
    
    gpt_file = 'gpt4_responses.csv'
    gpt_labels = pd.read_csv(gpt_file)
    gpt_texts = set(gpt_labels['text'].values.tolist())
    response_gpt_labels = gpt_labels.groupby('text')
    
    gpt_pred = []
    for text in gt_data['old_text'].values:
        label = 0
        if text in gpt_texts:
            response = response_gpt_labels.get_group(text)['responses'].values[0]
            if 'Yes' in response[:4]:
                label = 1
            elif 'No' in response[:4]:
                label = 0
        else:
          print('error')
        gpt_pred.append(label)
    
    gt_data['gpt4_pred'] = gpt_pred
    
    gt_xy =gt_data[['embeddings','hazard']].replace([None],np.nan).dropna() # ,'gpt_pred','gpt_soc_pred','gpt4_pred','gpt_lib_pred'
    X = np.array([v.astype('float32') for v in gt_xy['embeddings'].values])
    y = gt_xy[['hazard','hazard']].values#np.array([round(l) for l in gt_xy[['hazard']].values]) # ,'gpt_pred','gpt_soc_pred','gpt4_pred','gpt_lib_pred'
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.9, random_state=42)
    y_train = y_train[:,0].round().reshape(-1,1)
    y_test = y_test[:,0].round().reshape(-1,1)


    def build_model(nx, layers, activations, lambtha, keep_prob):
        #Function that builds a neural network with the Keras library
        #Args:
        #  nx is the number of input features to the network
        #  layers is a list containing the number of nodes in each layer of the
        #  network
        #  activations is a list containing the activation functions used for
        #  each layer of the network
        #  lambtha is the L2 regularization parameter
        #  keep_prob is the probability that a node will be kept for dropout
        #Returns: the keras model
        inputs = tf.keras.Input(shape=(nx,))
        regularizer = tf.keras.regularizers.l2(float(lambtha))
        output = tf.keras.layers.Dense(layers[0],
                                activation=activations[0],
                                kernel_regularizer=regularizer)(inputs)
        hidden_layers = range(len(layers))[1:]
        for i in hidden_layers:
            dropout = tf.keras.layers.Dropout(1 - float(keep_prob))(output)
            output = tf.keras.layers.Dense(layers[i], activation=activations[i],
                                    kernel_regularizer=regularizer)(dropout)
        model = tf.keras.Model(inputs, output)
        return model
    
    def optimize_model(network, alpha, beta1, beta2):
        #Function that sets up Adam optimization for a keras model with categorical
        #crossentropy loss and accuracy metrics
        #Args:
        #network is the model to optimize
        #alpha is the learning rate
        #beta1 is the first Adam optimization parameter
        #beta2 is the second Adam optimization parameter
        #Returns: None
        Adam = tf.keras.optimizers.Adam(learning_rate=float(alpha),
                                 beta_1=float(beta1),
                                 beta_2=beta2)
        network.compile(optimizer=Adam,
                        loss="categorical_crossentropy",
                        metrics=['accuracy', tf.keras.metrics.AUC()])
    
    def train_model(network, train_data, labels, batch_size, epochs,
                    validation_data=None, early_stopping=False,
                    patience=0, learning_rate_decay=False,
                    alpha=0.1, decay_rate=1, filepath=None,
                    verbose=False, shuffle=False):
        #Function That trains a model using mini-batch gradient descent
        #Args:
        #network is the model to train
        #data is a numpy.ndarray of shape (m, nx) containing the input data
        #labels is a one-hot numpy.ndarray of shape (m, classes) containing
        #the labels of data
        #batch_size is the size of the batch used for mini-batch gradient descent
        #epochs is the number of passes through data for mini-batch gradient descent
        #validation_data is the data to validate the model with, if not None    
        
        def learning_rate_decay(epoch):
            #"""Function tha uses the learning rate"""
            alpha_0 = alpha / (1 + (decay_rate * epoch)) 
            return alpha_0
        
        callbacks = []
        if validation_data:
            if early_stopping:
                early_stop = tf.keras.callbacks.EarlyStopping(patience=patience)
                callbacks.append(early_stop)
            if learning_rate_decay:
                decay = tf.keras.callbacks.LearningRateScheduler(learning_rate_decay,
                                                          verbose=verbose)
                callbacks.append(decay)
        if filepath:
            print(filepath)
            save = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True)
            callbacks.append(save)
        train = network.fit(x=train_data,
                            y=labels,
                            batch_size=int(batch_size),
                            epochs=epochs,
                            validation_data=validation_data,
                            callbacks=callbacks,
                            verbose=False,
                            shuffle=shuffle)
        return train
    
    def object_function(x):
            #Function that set hyperparameters of a keras network:
            #Args: X is a vector conating the parameter to optimized and trained
            #    lambtha is the L2 regularization parameter
            #    keep_prob is the probability that a node will be kept for dropout
            #    alpha is the learning rate in Adam optimizer
            #    beta1 is the first Adam optimization parameter
            #    batch_size is the size of the batch used for mini-batch  gradient
            #    descent
            #Returns the loss of the model
            def one_hot(Y, classes):
              #"""convert an array to a one-hot matrix"""
              m = Y.shape[0]
              one_hot = np.zeros((classes, m))
              one_hot[Y, np.arange(m)] = 1
              return one_hot.T
            # x is 5 dimentional vector with the parameter we want to optimize
            lambtha = x[:, 0]
            keep_prob = x[:, 1]
            alpha = x[:, 2]
            beta1 = x[:, 3]
            batch_size = x[:, 4]
            # Building the model using Keras library
            network = build_model(embedding_dim, [256, 256, 1], ['relu', 'relu', 'softmax'],
                                  lambtha, keep_prob)
            # Optimizing the model using adam optimizer
            beta2 = 0.999
            optimize_model(network, alpha, beta1, beta2)
            # Training the model using early stopping and saving the best modle
            # in bayes_opt.txt'
            epochs = 1000
            random_indices = list(range(len(X_train)))
            random.shuffle(random_indices)
            train_ind = random_indices[:int(0.8*len(X_train))]
            valid_ind = random_indices[int(0.8*len(X_train)):]
            X_train_i = X_train[train_ind]
            Y_train_i = y_train[train_ind]
            X_valid = X_train[valid_ind]
            Y_valid = y_train[valid_ind]
            history = train_model(network, X_train_i, Y_train_i, batch_size, epochs,
                                  validation_data=(X_valid, Y_valid),
                                  early_stopping=True, patience=3,
                                  learning_rate_decay=True)
            return (history.history['val_loss'][-1])
    
    
    # Setting the bounds of network parameter for the bayeyias optimizatio
    bounds = [{'name': 'lambtha', 'type': 'continuous','domain': (0.00005, 0.005)},
                {'name': 'keep_prob', 'type': 'continuous','domain': (0.05, 0.95)},
                {'name': 'alpha', 'type': 'continuous','domain': (0.0001, 0.005)},
                {'name': 'beta1', 'type': 'continuous', 'domain': (0.9, 0.999)},
                {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 128)}]
    
    # Creating the GPyOpt method using Bayesian Optimizatio
    my_Bayes_opt = GPyOpt.methods.BayesianOptimization(object_function,
                                                       domain=bounds)
    
    #Stop conditions
    max_time  = None
    max_iter  = 30
    tolerance = 1e-8
    
    #Running the method
    my_Bayes_opt.run_optimization(max_iter = max_iter,
                                  max_time = max_time,
                                  eps = tolerance)
    
    print("Value of (x,y) that minimises the objective:"+str(my_Bayes_opt.x_opt))
    for c,v in zip(['lambtha','keep_prob','alpha','beta1','batch_size'],my_Bayes_opt.x_opt):
        print([c,v])
    
    print("Minimum value of the objective: "+str(my_Bayes_opt.fx_opt))
    def train_model(x):
      print(x)
      batch_size,num_layers,l2_lambda,relu_alpha,dropout_rate = x[0]
      num_layers = int(num_layers)
      batch_size = int(batch_size)
      embedding_normalizer = Normalization(input_shape=[embedding_dim,], axis=None)
      embedding_normalizer.adapt(X_train)
      def modeling(l2_lambda, relu_alpha, dropout_rate,num_layers):
          prev_dim = int(embedding_dim)
          model = Sequential([embedding_normalizer])
          for i in range(num_layers):
              if prev_dim <=2:
                  break
              model.add(Dense(int(prev_dim/3),kernel_regularizer=regularizers.l2(l2_lambda)))
              model.add(LeakyReLU(relu_alpha)) #alpha=negative coefficient for the slope
              model.add(BatchNormalization())
              model.add(Dropout(dropout_rate))
              prev_dim = int(prev_dim/3)    
          model.add(Dense(1, activation=tf.keras.activations.sigmoid))
          model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_crossentropy',tf.keras.metrics.AUC()])
          return model
    
      es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5) #min_delta=1 #baseline=26
    
    
      m = modeling(l2_lambda, relu_alpha, dropout_rate,num_layers)
      history = m.fit(X_train, y_train, validation_split=0.1, epochs=1000, batch_size=batch_size, verbose=2,callbacks=[es])
      return (history.history['val_loss'][-1])
    
    bounds = [{'name': 'dropout_rate', 'type': 'continuous','domain': (0.1, 0.9)},
                {'name': 'relu_alpha', 'type': 'continuous','domain': (0.0, 0.2)},
                {'name': 'l2_lambda', 'type': 'continuous','domain': (0.0001, 0.005)},
                {'name': 'num_layers', 'type': 'discrete', 'domain': (1,2,3,4)},
                {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32,64)}]
    
    # Creating the GPyOpt method using Bayesian Optimizatio
    my_Bayes_opt = GPyOpt.methods.BayesianOptimization(train_model,
                                                       domain=bounds)
    #Stop conditions
    max_time  = None
    max_iter  = 30
    tolerance = 1e-8
    
    #Running the method
    my_Bayes_opt.run_optimization(max_iter = max_iter,
                                  max_time = max_time,
                                  eps = tolerance)
    print("Value of (x,y) that minimises the objective:"+str(my_Bayes_opt.x_opt))
    for c,v in zip([b['name'] for b in bounds],my_Bayes_opt.x_opt):
        print([c,v])
    print([b['name'] for b in bounds])
    print(list(my_Bayes_opt.x_opt))
    print("Minimum value of the objective: "+str(my_Bayes_opt.fx_opt))





    search_space ={
    'n_estimators':Integer( 10, 150),
    'max_depth':Integer( 5, 50),
    'min_samples_split':Integer( 2,20),
    'max_features':Categorical(['sqrt','log2',None]),
    'class_weight':Categorical(['balanced',None]),
    'ccp_alpha':Real(0.0,0.01)#,
    }
    #search_space = [n_estimators,max_depth,min_samples_split,max_features,class_weight,ccp_alpha,monotonic_cst]
    
    
    optimizer = BayesSearchCV(
        estimator=RandomForestClassifier(),
        search_spaces=search_space,
        scoring=None,
        cv=5,
        n_iter=10,
        return_train_score=False,
        n_jobs=-1
    )
    
    optimizer.fit(X_train, y_train)
    rf_best_hyperparameters = optimizer.best_params_
    best_score = optimizer.best_score_
    print([rf_best_hyperparameters,best_score])


    kwargs = {}
    for key,value in rf_best_hyperparameters.items():
      kwargs[key] = value
    print(kwargs)
    clf = RandomForestClassifier(**kwargs)
    clf.fit(X_train, y_train)
    pred_prob = clf.predict_proba(X_test)[:,1]
    pred = clf.predict(X_test)
    
    auc = roc_auc_score(y_test, pred_prob)
    f1=f1_score(y_test, pred)
    print([auc,f1])
    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    pred_prob = clf.predict_proba(X_test)[:,1]
    pred = clf.predict(X_test)
    
    auc = roc_auc_score(y_test, pred_prob)
    f1=f1_score(y_test, pred)
    print([auc,f1])
    search_space ={
    'C':Real(0.01,10),
    'kernel':Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
    'degree':Integer(1,4),
    'gamma':Categorical(['auto','scale']),
    'shrinking':Categorical([False,True]),
    'class_weight':Categorical(['balanced',None]),
    }
    
    
    optimizer = BayesSearchCV(
        estimator=SVC(probability=True),
        search_spaces=search_space,
        scoring=None,#'roc_auc',
        cv=5,
        n_iter=10,
        return_train_score=False,
        n_jobs=-1
    )
    
    optimizer.fit(X_train, y_train)
    svc_best_hyperparameters = optimizer.best_params_
    best_score = optimizer.best_score_
    print([svc_best_hyperparameters,best_score])
    kwargs = {}
    for key,value in svc_best_hyperparameters.items():
      kwargs[key] = value
    
    clf = SVC(probability=True,**kwargs)
    clf.fit(X_train, y_train)
    pred_prob = clf.predict_proba(X_test)[:,1]
    pred = clf.predict(X_test)
    
    auc = roc_auc_score(y_test, pred_prob)
    f1=f1_score(y_test, pred)
    print([auc,f1])
    
    
    clf = SVC(probability=True)
    clf.fit(X_train, y_train)
    pred_prob = clf.predict_proba(X_test)[:,1]
    pred = clf.predict(X_test)
    
    auc = roc_auc_score(y_test, pred_prob)
    f1=f1_score(y_test, pred)
    print([auc,f1])
    search_space ={
    'n_estimators':Integer( 10, 100),
    'max_depth':Integer( 5, 50),
    'max_leaves':Integer( 20,200),
    'max_bin':Integer( 2,200),
    'tree_method':Categorical(['auto', 'exact', 'approx', 'hist']),
    'importance_type':Categorical(['gain','weight','cover','total_gain','total_cover'])
    }
    
    
    optimizer = BayesSearchCV(
        estimator=XGBClassifier(),
        search_spaces=search_space,
        scoring=None,#'roc_auc',
        cv=5,
        n_iter=10,
        return_train_score=False,
        n_jobs=-1
    )
    
    optimizer.fit(X_train, y_train)
    xgb2_best_hyperparameters = optimizer.best_params_
    best_score = optimizer.best_score_
    print([xgb2_best_hyperparameters,best_score])
    kwargs = {}
    for key,value in xgb2_best_hyperparameters.items():
      kwargs[key] = value
    
    clf = XGBClassifier(**kwargs)
    clf.fit(X_train, y_train)
    pred_prob = clf.predict_proba(X_test)[:,1]
    pred = clf.predict(X_test)
    
    auc = roc_auc_score(y_test, pred_prob)
    f1=f1_score(y_test, pred)
    print([auc,f1])
    
    
    
    metrics = {'NN_auc':[],'NN_f1':[],'RF_auc':[],'RF_f1':[],'SVM_auc':[],'SVM_f1':[],'XGB_auc':[],'XGB_f1':[],'base_f1':[]}
    for ii in range(50):
        boot_indices = np.random.randint(0,len(X_test),len(X_test))
        X_boot = X_test[boot_indices]
        y_boot = y_test[boot_indices]
        y_boot = y_boot[:,0].round().reshape(-1,1)
    
        embedding_normalizer = Normalization(input_shape=[embedding_dim,], axis=None)
        embedding_normalizer.adapt(X_train)
        dropout_rate, relu_alpha, l2_lambda, num_layers, batch_size = my_Bayes_opt.x_opt
        num_layers = int(num_layers)
        batch_size = int(batch_size)
        if True:
          def modeling(l2_lambda, relu_alpha, dropout_rate,num_layers):
            prev_dim = int(embedding_dim)
            model = Sequential([embedding_normalizer])
            for i in range(num_layers):
                if prev_dim <=2:
                    break
                model.add(Dense(int(prev_dim/3),kernel_regularizer=regularizers.l2(l2_lambda)))
                model.add(LeakyReLU(relu_alpha)) #alpha=negative coefficient for the slope
                model.add(BatchNormalization())
                model.add(Dropout(dropout_rate))
                prev_dim = int(prev_dim/3)    
            model.add(Dense(1, activation=tf.keras.activations.sigmoid))
            model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_crossentropy',tf.keras.metrics.AUC()])
            return model
          if True:
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5) #min_delta=1 #baseline=26
            m = modeling(l2_lambda, relu_alpha, dropout_rate,num_layers)
            history = m.fit(X_train, y_train, validation_split=0.1, epochs=1000, batch_size=32, verbose=2,callbacks=[es]) #callbacks=[es])
            y_pred = m.predict(X_boot)
            metrics['NN_auc'].append(roc_auc_score(y_boot, y_pred))
            metrics['NN_f1'].append(f1_score(y_boot,y_pred.round()))
    
          kwargs = {}
          for key,value in rf_best_hyperparameters.items():
            kwargs[key] = value
          clf = RandomForestClassifier(**kwargs)
          clf.fit(X_train, y_train)
          pred_prob = clf.predict_proba(X_boot)[:,1]
          pred = clf.predict(X_boot)
          auc = roc_auc_score(y_boot, pred_prob)
          f1=f1_score(y_boot, pred)
          metrics['RF_auc'].append(auc)
          metrics['RF_f1'].append(f1)
          kwargs = {'probability':True}
          for key,value in svc_best_hyperparameters.items():
            kwargs[key] = value
          clf = SVC(**kwargs)
          clf.fit(X_train, y_train)    
          pred_prob = clf.predict_proba(X_boot)[:,1]
          pred = clf.predict(X_boot)
          auc = roc_auc_score(y_boot, pred_prob)
          f1=f1_score(y_boot, pred)
          metrics['SVM_auc'].append(auc)
          metrics['SVM_f1'].append(f1)
          kwargs = {}
          for key,value in xgb2_best_hyperparameters.items():
            kwargs[key] = value
          clf = XGBClassifier(**kwargs)
          clf.fit(X_train, y_train)
    
          pred_prob = clf.predict_proba(X_boot)[:,1]
          auc = roc_auc_score(y_boot, pred_prob)
    
          pred = clf.predict(X_boot)
          f1=f1_score(y_boot, pred)
          metrics['XGB_auc'].append(auc)
          metrics['XGB_f1'].append(f1)
          print([auc,f1])
    
          f1_base=f1_score(y_boot,[1]*len(y_boot))
          metrics['base_f1'].append(f1_base)
    
    pd.DataFrame(metrics).to_csv('metrics_gpt_vs_gpt_s_new_'+embed_model.replace('/','_')+'_correct_split_11.csv',index=False)
