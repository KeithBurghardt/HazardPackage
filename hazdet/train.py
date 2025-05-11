# Imports
# Basic
# !pip install demoji
# !pip install -U sentence-transformers
# !pip install scikit-learn
# !pip install scikit-optimize


import pandas as pd
import numpy as np
import random
import pickle as pk
from scipy.stats import beta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import demoji
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Normalization
from tensorflow.keras.optimizers import Adam
from keras.metrics import AUC
from xgboost import XGBClassifier


def extract_qualtrics_data(file='Tweet annotation.csv'):
    # make sure we only include annotators who completed full survey
    progress_pct = 100
    # find all data where humans completed the task and took > 3 minutes, 20 seconds (200 seconds)
    duration_seconds = 200

    data = pd.read_csv(file,low_memory=False,lineterminator='\n')
    
    # make columns readable
    data.columns = [c.replace('_1','') if not ('_' in c and 'Q' not in c) else 'Q'+c.replace('_1','') for c in data.columns]
    # this question checks if someone is paying attention; everyone chose "4" so it is not needed
    data = data.drop(['Q11119'],axis=1)
    cols = data.columns
    # where are the questions for each text located
    pos_qs = [ii for ii,c in enumerate(cols) if 'Q' in c]
    # number of questions per annotations
    num_qs = 11
    # where do the questions start (starting from the first and ending at the 10th annotation)
    q_start_pos = [22-num_qs]+pos_qs[5::11]
    # here we extract the exact questions
    col_text = {}
    for c in cols:
        col_text[c] = data.iloc[0][c]
    questions = [col_text[c] for c in cols[q_start_pos[1]:q_start_pos[1]+11]]
    # demographics
    demogs = [col_text[c] for c in cols if 'QD' in c]
    # all of the key data: text, questions, demographics per annotator
    annotations = {'text':[]}
    for q in questions:
        annotations[q] = []
    for d in demogs:
        annotations[d] = []

    # data cleaning
    data = data[12:]
    data = data.loc[data['Progress'].astype(float).values==progress_pct,]
    data = data.loc[data['Duration (in seconds)'].astype(float).values>duration_seconds,]
    return data,annotations

def reshape_qualtrics_data(data,annotations):
    
    # for each row of this file, find the exact text humans annotated
    for ii,row in data.iterrows():
        if ii > 0:
            # all the tweets are those whose elements are not null and not questions
            texts = [col_text[c] for null,c in zip(row.isnull()[q_start_pos[0]+num_qs:],cols[q_start_pos[0]+num_qs:]) if not null and 'Q' not in c and c in col_text.keys()][:-4]
            line_annots = {}
            # for each text,
            for q1,text in zip(q_start_pos[1:],texts):
                # positions of all the questions associated with the text
                q_cols = cols[q1:q1+num_qs]
                # add tweet
                line_annots['text'] = [text]
                # add questions
                for q,c in zip(questions,q_cols):
                    line_annots[q]=[row[c]]
                for c in cols:
                    # demographic questions
                    if 'QD' in c:
                        line_annots[col_text[c]]=[row[c]]
            # if we have found all the data
            if set(list(line_annots.keys()))== set(list(annotations.keys())):
                # append text, questions, demographics
                for q1,text in zip(q_start_pos[1:],texts):
                    q_cols = cols[q1:q1+num_qs]
                    annotations['text'].append(text)
                    for q,c in zip(questions,q_cols):
                        annotations[q].append(row[c])
                    for c in cols:
                        if 'QD' in c:
                            annotations[col_text[c]].append(row[c])
                lens= []
                for key in annotations.keys():
                    lens.append(len(annotations[key]))
    return annotations

def clean_qualtrics_data(file='Tweet annotation.csv'):
    data,annotations = extract_qualtrics_data(file)
    annotations = reshape_qualtrics_data(data,annotations)
    annotations = pd.DataFrame(annotations)
    hazard_col = 'Does the tweet describe a hazard (something that could impose harm or other costs on the author of the tweet or on others)?'
    benefit_col = 'Does the tweet describe a benefit (something that provides resources, opportunities, or other good things to the author of the tweet or to others)?'
    return annotations,hazard_col,benefit_col

def create_features(file='Tweet annotation.csv'):
    # extract resshaped data
    annotations,hazard_col,benefit_col = clean_qualtrics_data(file)
    text_haz_ben=annotations[['text',hazard_col,benefit_col]]
    text_haz_ben[hazard_col] = text_haz_ben[hazard_col].replace('Yes',1).replace('No',0).dropna()
    text_haz_ben[benefit_col] = text_haz_ben[benefit_col].replace('Yes',1).replace('No',0).dropna()
    unique_text = text_haz_ben['text'].drop_duplicates()
    haz_ben_annots = text_haz_ben.groupby('text')
    GT_labels = {'text':[],'hazard':[],'benefit':[],'old_text':[]}
    for t in unique_text:
        annots = haz_ben_annots.get_group(t)
        # require at least 2 annotations
        if len(annots) <= 2: continue
        ben = annots[benefit_col].values.sum()/len(annots)
        haz = annots[hazard_col].values.sum()/len(annots)
        all_replace={}
        replaced_text = []
        # replace emojis
        replace = demoji.findall(t)
        new_t = t
        for word, initial in replace.items():
            new_t = new_t.replace(word, initial)
        GT_labels['old_text'].append(t)
        GT_labels['text'].append(new_t)
        GT_labels['hazard'].append(haz)
        GT_labels['benefit'].append(ben)
    GT_labels = pd.DataFrame(GT_labels)
    #Sentences are encoded by calling model.encode()
    model = SentenceTransformer('stsb-xlm-r-multilingual')#'all-mpnet-base-v2')
    embeddings = model.encode(GT_labels['text'].values)
    GT_labels['embeddings'] = [e for e in embeddings]
    GT_labels.sample(frac=1,replace=False)# mix up the rows
    gt_xy =GT_labels[['embeddings','hazard','benefit']].replace([None],np.nan).dropna()
    X = np.array([v.astype('float32') for v in gt_xy['embeddings'].values])
    y = gt_xy[['hazard','benefit']].values.round()
    return X,y

def hyperparameter_tune_model(X,y,search_space,model):
    import skopt
    from skopt.space import Real, Integer,Categorical
    from skopt import BayesSearchCV
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.9, random_state=42)
    y_train = y_train[:,0].round().reshape(-1,1)
    y_test = y_test[:,0].round().reshape(-1,1)
    optimizer = BayesSearchCV(
    estimator=model,
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
    return rf_best_hyperparameters,best_score


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

    inputs = keras.Input(shape=(nx,))
    regularizer = regularizers.l2(float(lambtha))

    output = Dense(layers[0],
                            activation=activations[0],
                            kernel_regularizer=regularizer)(inputs)

    hidden_layers = range(len(layers))[1:]

    for i in hidden_layers:
        dropout = keras.layers.Dropout(1 - float(keep_prob))(output)
        output = Dense(layers[i], activation=activations[i],
                                kernel_regularizer=regularizer)(dropout)

    model = keras.Model(inputs, output)

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

    adam = Adam(learning_rate=float(alpha),
                             beta_1=float(beta1),
                             beta_2=beta2)

    network.compile(optimizer=adam,
                    loss="categorical_crossentropy",
                    metrics=['accuracy', AUC()])
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
            early_stop = EarlyStopping(patience=patience)
            callbacks.append(early_stop)

        if learning_rate_decay:
            decay = keras.callbacks.LearningRateScheduler(learning_rate_decay,
                                                      verbose=verbose)
            callbacks.append(decay)


    if filepath:
        print(filepath)
        save = keras.callbacks.ModelCheckpoint(filepath, save_best_only=True)
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
        network = build_model(768, [256, 256, 1], ['relu', 'relu', 'softmax'],
                              lambtha, keep_prob)

        # Optimizing the model using adam optimizer
        beta2 = 0.999
        optimize_model(network, alpha, beta1, beta2)

        # Training the model using early stopping and saving the best modle
        # in bayes_opt.txt'
        epochs = 100
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

def hyperparameter_tune_nn(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.9, random_state=42)
    y_train = y_train[:,0].round().reshape(-1,1)
    y_test = y_test[:,0].round().reshape(-1,1)

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

    nn_hyperparameters = [(c,v) for c,v in zip(['lambtha','keep_prob','alpha','beta1','batch_size'],my_Bayes_opt.x_opt)]
    return nn_hyperparameters

def hyperparameter_tune_all_models(file):
    X,y = create_features(file)
    search_space ={
        'n_estimators':Integer( 10, 150),
        'max_depth':Integer( 5, 50),
        'min_samples_split':Integer( 2,20),
        'max_features':Categorical(['sqrt','log2',None]),
        'class_weight':Categorical(['balanced',None]),
        'ccp_alpha':Real(0.0,0.01)#,
    }
    rf_best_hyperparameters,_ = hyperparameter_tune(X,y,search_space,RandomForestClassifier())

    search_space ={
    'C':Real(0.01,10),
    'kernel':Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
    'degree':Integer(1,4),
    'gamma':Categorical(['auto','scale']),
    'shrinking':Categorical([False,True]),
    'class_weight':Categorical(['balanced',None]),
    } 
    svc_best_hyperparameters,_ = hyperparameter_tune(X,y,search_space,SVC(probability=True))
    
    search_space ={
    'n_estimators':Integer( 10, 100),
    'max_depth':Integer( 5, 50),
    'max_leaves':Integer( 20,200),
    'max_bin':Integer( 2,200),
    'tree_method':Categorical(['auto', 'exact', 'approx', 'hist']),
    'gamma':Real(0.0,0.1),
    'colsample_bytree':Real(0.1,1.0),
    'colsample_bylevel':Real(0.1,1.0),
    'reg_alpha':Real(0.0,10.0), 
    'reg_lambda':Real(0.0,10.0),
    'importance_type':Categorical(['gain','weight','cover','total_gain','total_cover'])
    }
    xgb_best_hyperparameters,_ = hyperparameter_tune(X,y,search_space,XGBClassifier())
    nn_best_hyperparameters = hyperparameter_tune_nn(X,y)
    params = [rf_best_hyperparameters,svc_best_hyperparameters,xgb_best_hyperparameters,nn_best_hyperparameters]
    # save the best model...
    train_best_model(X,y,params)


def train_nn(nn_best_hyperparameters,X_train, y_train,embedding_dim=768):
    embedding_normalizer = Normalization(input_shape=[embedding_dim,], axis=None)
    embedding_normalizer.adapt(X_train)
    dropout_rate,relu_alpha,l2_lambda,num_layers,batch_size = nn_best_hyperparameters
    num_layers = int(num_layers)
    batch_size = int(batch_size)
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
        model.add(Dense(1, activation=keras.activations.sigmoid))
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_crossentropy',AUC()])
        return model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5) #min_delta=1 #baseline=26
    m = modeling(l2_lambda, relu_alpha, dropout_rate,num_layers)
    history = m.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=32, verbose=2,callbacks=[es]) #callbacks=[es])
    return m

def predict_model (model,model_params,X,y,ii):
    random_state = 999
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.9, random_state=random_state)
    y_train = y_train[:,0].round().reshape(-1,1)
    # random seed
    numpy.random.seed(ii*314159)
    boot_indices = np.random.randint(0,len(X_test),len(X_test))
    X_boot = X_test[boot_indices]
    y_boot = y_test[boot_indices]    
    if model == 'NN':
        clf = train_nn(model_params,X_train, y_train)
        y_pred = clf.predict(X_boot)

    elif model == 'RF':
        kwargs = {key:value for key,value in model_params.items()}
        clf = RandomForestClassifier(**kwargs)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_boot)[:,1]

    elif model == 'SVC':
        kwargs = {key:value for key,value in svc_best_hyperparameters.items()}
        kwargs['probability'] = True
        clf = SVC(**kwargs)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_boot)[:,1]
    elif model == 'XGB':
        kwargs = {key:value for key,value in xgb2_best_hyperparameters.items()}
        clf = XGBClassifier(**kwargs)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_boot)[:,1]
    return y_pred,y_boot


def eval_best_model(X,y,params,num_evals = 50,eval_metric='roc_auc'):
    rf_best_hyperparameters,svc_best_hyperparameters,xgb_best_hyperparameters,nn_best_hyperparameters = params
    # X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.9, random_state=999)#42)
    # y_train = y_train[:,0].round().reshape(-1,1)

    #metrics = {'NN_auc':[],'NN_f1':[],'RF_auc':[],'RF_f1':[],'SVM_auc':[],'SVM_f1':[],'XGB_auc':[],'XGB_f1':[],'gpt_auc':[],'gpt_f1':[],'base_f1':[],'gpt_auc':[],'gpt_f1':[],'gpt_soc_auc':[],'gpt_soc_f1':[],'gpt_lib_auc':[],'gpt_lib_f1':[],'gpt4_auc':[],'gpt4_f1':[]}
    best_model = ['',0,0]
    performance = {}
    for model in ['NN','SVC','RF','XGB']:
        performance_metric_boot = []
        for ii in range(num_evals):
            y_pred,y_boot = predict_model (model,model_params,X,y,ii) 
            if eval_metric ==  'roc_auc':
                performance = roc_auc_score(y_boot, y_pred)
            elif eval_metric == 'f1':
                performance = f1_score(y_boot,y_pred.round())
            else:
                raise Exception('ERROR: Evaluation metric not recognized.')
            performance_metric_boot.append(performance)
        mean_performance = np.mean(performance_metric_boot)
        std_performance = np.std(performance_metric_boot)
        performance[model] = [mean_performance,std_performance]
        if mean_performance > best_model[1]:
            best_model[0] = model

    return best_model[0],performance


def train_best_model(X,y,params):
    # find the best model
    best_model,performance = eval_best_model(X,y,params,num_evals = 50,eval_metric='roc_auc')
    if best_model == 'NN':
        clf = train_nn(model_params,X, y)
        y_pred = clf.predict(X_boot)
            
    elif best_model == 'RF':
        kwargs = {key:value for key,value in model_params.items()}
        clf = RandomForestClassifier(**kwargs)
        clf.fit(X, y)
    
    elif best_model == 'SVC':
        kwargs = {key:value for key,value in svc_best_hyperparameters.items()}
        kwargs['probability'] = True
        clf = SVC(**kwargs)
        clf.fit(X, y)
    elif best_model == 'XGB':
        kwargs = {key:value for key,value in xgb2_best_hyperparameters.items()}
        clf = XGBClassifier(**kwargs)
        clf.fit(X, y)
    if best_model == 'NN':
        clf.save('finalized_model_NN.sav')
        return

    # otherwise...
    import pickle as pk
    filename = 'finalized_model_'+best_model+'.sav'
    pk.dump(clf, open(filename, 'wb'))
    return
