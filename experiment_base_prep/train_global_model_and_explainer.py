#!/usr/bin/env python

import sys, os,  pickle, time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from IPython.display import display
from my_util import *
from lime.lime.lime_tabular import LimeTabularExplainer
from tqdm import tqdm
import time

from pyexplainer.pyexplainer_pyexplainer import *

sys.path.append(os.path.abspath('../'))
from mohit_base_algorithm.pyexplainer_pyexplainer import *

import warnings
warnings.filterwarnings("ignore")

data_path = './dataset/'
result_dir = './eval_result/'
dump_dataframe_dir = './prediction_result/'
exp_dir = './explainer_object/'
d_dir = './synthetic_data/'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    
if not os.path.exists(dump_dataframe_dir):
    os.makedirs(dump_dataframe_dir)
    
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

if not os.path.exists(d_dir):
    os.makedirs(d_dir)

def train_global_model(proj_name, x_train,y_train, global_model_name = 'RF'):
    
    smt = SMOTE(k_neighbors=5, random_state=42, n_jobs=24)
    new_x_train, new_y_train = smt.fit_resample(x_train, y_train)
    
    if global_model_name == 'RF':
        global_model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=24)
    elif global_model_name == 'LR':
        global_model = LogisticRegression(random_state=0, n_jobs=24)
        
    global_model.fit(new_x_train, new_y_train)
    pickle.dump(global_model, open(proj_name+'_'+global_model_name+'_global_model.pkl','wb'))

def get_correctly_predicted_defective_commit_indices(proj_name, global_model_name, x_test, y_test):

    prediction_df_dir = dump_dataframe_dir+proj_name+'_'+global_model_name+'_prediction_result.csv'
    correctly_predict_df_dir = dump_dataframe_dir+proj_name+'_'+global_model_name+'_correctly_predict_as_defective.csv'
    
    if not os.path.exists(prediction_df_dir) or not os.path.exists(correctly_predict_df_dir):
        global_model = pickle.load(open(proj_name+'_'+global_model_name+'_global_model.pkl','rb'))

        pred = global_model.predict(x_test)
        defective_prob = global_model.predict_proba(x_test)[:,1]

        prediction_df = x_test.copy()
        prediction_df['pred'] = pred
        prediction_df['defective_prob'] = defective_prob
        prediction_df['defect'] = y_test

        correctly_predict_df = prediction_df[(prediction_df['pred']==1) & (prediction_df['defect']==1)]

        prediction_df.to_csv(prediction_df_dir)
        correctly_predict_df.to_csv(correctly_predict_df_dir)
    
    else:
        prediction_df = pd.read_csv(prediction_df_dir)
        correctly_predict_df = pd.read_csv(correctly_predict_df_dir)
        
        prediction_df = prediction_df.set_index('commit_id')
        correctly_predict_df = correctly_predict_df.set_index('commit_id')
        
    return correctly_predict_df.index

def train_global_model_runner(proj_name, global_model_name):
    x_train, x_test, y_train, y_test = prepare_data(proj_name, mode = 'all')

    train_global_model(proj_name, x_train, y_train,global_model_name)
    print('train {} of {} finished'.format(global_model_name, proj_name))

def create_explainer(proj_name, global_model_name, x_train, x_test, y_train, y_test, df_indices, debug=False):
    
    save_dir = os.path.join(exp_dir,proj_name,global_model_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    global_model = pickle.load(open(proj_name+'_'+global_model_name+'_global_model.pkl','rb'))

    indep = x_test.columns
    dep = 'defect'
    class_label = ['clean', 'defect']
    

    #################################
    # for authors apporach
    pyExp = PyExplainer(x_train, y_train, indep, dep, global_model, class_label)

    # for baseline
    # note: 6 is index of 'self' feature
    lime_explainer = LimeTabularExplainer(x_train.values, categorical_features=[6],
                                      feature_names=indep, class_names=class_label, 
                                      random_state=0)
    #################################

    # for my testing 
    mBase = MohitBase(x_train, y_train, indep, dep, global_model, class_label)


    feature_df = x_test.loc[df_indices]
    test_label = y_test.loc[df_indices]

    mBase_time = []
    pyExp_time = []
    lime_time= []
    
    def _prepareExplainerObj(usePreTrainedData=False, preTrainedData=None, search_function='ctgan') : 
        if debug : 
            print("\t..starting Mohit-base with search function = ", search_function)

        start = time.time()   
        mBase_obj = mBase.explain(X_explain,
                                   y_explain,
                                   modelType = "nc", 
                                   cv = 1,
                                   search_function = search_function,
                                   usePreTrainedData = usePreTrainedData,
                                   preTrainedData = preTrainedData,
                                   debug=debug)
        end = time.time()
        elapsed = end - start
        
        if debug : 
            print("\t..done Mohit-base with search function = ", search_function)

        # add row index to the object 
        mBase_obj['commit_id'] = row_index
        # add time stamp to the object
        mBase_obj['time'] = elapsed

        return mBase_obj

    for i in tqdm(range(0,len(feature_df))):
        X_explain = feature_df.iloc[[i]]
        y_explain = test_label.iloc[[i]]

        row_index = str(X_explain.index[0])

        if debug : 
            print("\nFor Row Index", row_index)


        #########   LIME  #############
        if debug : 
            print("\tstarting lime")
        X_explain = feature_df.iloc[i] # to prevent error in LIME
        start = time.time()
        exp, synt_inst, synt_inst_for_local_model, selected_feature_indices, local_model = lime_explainer.explain_instance(X_explain, global_model.predict_proba, num_samples=5000)

        lime_obj = {}
        lime_obj['rule'] = exp
        lime_obj['synthetic_instance_for_global_model'] = synt_inst
        lime_obj['synthetic_instance_for_lobal_model'] = synt_inst_for_local_model
        lime_obj['local_model'] = local_model
        lime_obj['selected_feature_indeces'] = selected_feature_indices
        lime_obj['commit_id'] = row_index
        end = time.time()
        elapsed = end - start
        lime_time.append(elapsed)
        # add time stamp to the object
        lime_obj['time'] = elapsed
        if debug : 
            print("\t..done lime")
        #################################


        
        ##########  PyExplainer ############
        if debug : 
            print("\tstarting pyexplainer")
        start = time.time()
        pyExp_obj = pyExp.explain(X_explain,
                                y_explain,
                                search_function = 'CrossoverInterpolation')
        end = time.time()
        elapsed = end - start
        pyExp_time.append(elapsed)
        if debug : 
            print("\t..done pyexplainer")

        # add row index to the object 
        pyExp_obj['commit_id'] = row_index
        # add time stamp to the object
        pyExp_obj['time'] = elapsed

        # because I don't want to change key name in another evaluation file
        pyExp_obj['local_model'] = pyExp_obj['local_rulefit_model']
        del pyExp_obj['local_rulefit_model']
        #################################
        
        preTrainedAllData = pickle.load(open(os.path.join(d_dir,proj_name,global_model_name,'syndata_'+row_index+'.pkl'),'rb'))

        # #################################
        mctgan_obj = _prepareExplainerObj(usePreTrainedData=True, preTrainedData=preTrainedAllData['ctgan'],search_function='ctgan') 
        mcopulagan_obj = _prepareExplainerObj(usePreTrainedData=True, preTrainedData=preTrainedAllData['copulagan'],search_function='copulagan')
        mtvae_obj = _prepareExplainerObj(usePreTrainedData=True, preTrainedData=preTrainedAllData['tvae'],search_function='tvae')
        mgcopula_obj = _prepareExplainerObj(usePreTrainedData=True, preTrainedData=preTrainedAllData['gcopula'],search_function='gcopula')
        mcrossinter_obj = _prepareExplainerObj(usePreTrainedData=True, preTrainedData=preTrainedAllData['crossoverinterpolation'],search_function='crossoverinterpolation')
        '''
        ################### When All Explainers Created ########## 
        # create all explainer object
        all_explainer = {'pyExplainer':pyExp_obj, 'LIME': lime_obj, 'MBase' : mBase_obj}
        
        # write the updated object. 
        pickle.dump(all_explainer, open(save_dir+'/all_explainer_'+row_index+'.pkl','wb'))
        ################### When All Explainers Created ########## 
        '''

        # '''
        ################### Loading Already Trained Files ########## 
        # load already trained file, from previous executions. 
        all_explainer = pickle.load(open(save_dir+'/all_explainer_'+row_index+'.pkl','rb'))

        # update the object
        all_explainer = {'pyexp':pyExp_obj, 
                         'LIME': lime_obj, 
                         'mctgan' : mctgan_obj, 
                         'mcopulagan' : mcopulagan_obj, 
                         'mtvae' : mtvae_obj, 
                         'mgcopula' : mgcopula_obj, 
                         'mcrossinter' : mcrossinter_obj, 
                         }
        
        # 'mcrossinter' : mcrossinter_obj, 

        
        # write the updated object. 
        pickle.dump(all_explainer, open(save_dir+'/all_explainer_'+row_index+'.pkl','wb'))
        ################### Loading Already Trained Files ########## 
        # '''

        if debug : 
            break

        # print('finished {}/{} commits'.format(str(i+1), str(len(feature_df))))

    # pickle.dump(mBase_time, open(save_dir+'/mBase_time'+'.pkl','wb'))
    # pickle.dump(pyExp_time, open(save_dir+'/pyExp_time'+'.pkl','wb'))
    # pickle.dump(lime_time, open(save_dir+'/lime_time'+'.pkl','wb'))

    # mBase.logBestParams()

def create_baseExplainer(proj_name, global_model_name, x_train, x_test, y_train, y_test, df_indices, debug=False):
    
    save_dir = os.path.join(exp_dir,proj_name,global_model_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    global_model = pickle.load(open(proj_name+'_'+global_model_name+'_global_model.pkl','rb'))

    indep = x_test.columns
    dep = 'defect'
    class_label = ['clean', 'defect']
    

    #################################
    # for baseline
    # note: 6 is index of 'self' feature
    lime_explainer = LimeTabularExplainer(x_train.values, categorical_features=[6],
                                      feature_names=indep, class_names=class_label, 
                                      random_state=0)
    #################################

    # for my testing 
    mBase = MohitBase(x_train, y_train, indep, dep, global_model, class_label)


    feature_df = x_test.loc[df_indices]
    test_label = y_test.loc[df_indices]

    mBase_time = []
    pyExp_time = []
    lime_time= []
    
    def _preparePyExplainerObj(usePreTrainedData=False, preTrainedData=None, modelType ='py', search_function='ctgan') : 
        if debug : 
            print("\t..starting Mohit-base with search function = ", search_function , ' and modelType = ', modelType)

        # print(type(X_explain))
        start = time.time()   
        mBase_obj = mBase.explain(X_explain,
                                   y_explain,
                                   modelType = modelType, 
                                   search_function = search_function,
                                   usePreTrainedData = usePreTrainedData,
                                   preTrainedData = preTrainedData,
                                   debug=debug)
        end = time.time()
        elapsed = end - start
        
        if debug : 
            print("\t..done Mohit-base with search function = ", search_function)

        # add row index to the object 
        mBase_obj['commit_id'] = row_index
        # add time stamp to the object
        mBase_obj['time'] = elapsed

        if modelType == 'py' : 
            mBase_obj['local_model'] = mBase_obj['local_rulefit_model']
        del mBase_obj['local_rulefit_model']

        return mBase_obj

    for i in tqdm(range(0,len(feature_df))):
        X_explain = feature_df.iloc[[i]]

        # print(type(X_explain))
        y_explain = test_label.iloc[[i]]

        row_index = str(X_explain.index[0])

        if debug : 
            print("\nFor Row Index", row_index)


        #########   LIME  #############
        if debug : 
            print("\tstarting lime")
        X_explain = feature_df.iloc[i] # to prevent error in LIME
        start = time.time()
        exp, synt_inst, synt_inst_for_local_model, selected_feature_indices, local_model = lime_explainer.explain_instance(X_explain, global_model.predict_proba, num_samples=5000)

        lime_obj = {}
        lime_obj['rule'] = exp
        lime_obj['synthetic_instance_for_global_model'] = synt_inst
        lime_obj['synthetic_instance_for_lobal_model'] = synt_inst_for_local_model
        lime_obj['local_model'] = local_model
        lime_obj['selected_feature_indeces'] = selected_feature_indices
        lime_obj['commit_id'] = row_index
        end = time.time()
        elapsed = end - start
        lime_time.append(elapsed)
        # add time stamp to the object
        lime_obj['time'] = elapsed
        if debug : 
            print("\t..done lime")
        #################################

        
        
        if debug : 
            print('done lime now towards further modules')
        preTrainedAllData = pickle.load(open(os.path.join(d_dir,proj_name,global_model_name,'syndata_'+row_index+'.pkl'),'rb'))

        X_explain = feature_df.iloc[[i]]
        
        if debug : 
            print('pretrained data set loaded. ')
        # #################################
        py_ctgan_obj = _preparePyExplainerObj(usePreTrainedData=True, preTrainedData=preTrainedAllData['ctgan'],modelType = 'py' , search_function='ctgan') 
        py_copulagan_obj = _preparePyExplainerObj(usePreTrainedData=True, preTrainedData=preTrainedAllData['copulagan'],modelType = 'py' ,search_function='copulagan')
        py_tvae_obj = _preparePyExplainerObj(usePreTrainedData=True, preTrainedData=preTrainedAllData['tvae'],modelType = 'py' ,search_function='tvae')
        py_gcopula_obj = _preparePyExplainerObj(usePreTrainedData=True, preTrainedData=preTrainedAllData['gcopula'],modelType = 'py' ,search_function='gcopula')
        py_crossinter_obj = _preparePyExplainerObj(usePreTrainedData=True, preTrainedData=preTrainedAllData['crossoverinterpolation'],modelType = 'py' ,search_function='crossoverinterpolation')
        '''
        ################### When All Explainers Created ########## 
        # create all explainer object
        all_explainer = {'pyExplainer':pyExp_obj, 'LIME': lime_obj, 'MBase' : mBase_obj}
        
        # write the updated object. 
        pickle.dump(all_explainer, open(save_dir+'/all_explainer_'+row_index+'.pkl','wb'))
        ################### When All Explainers Created ########## 
        '''

        # update the object
        all_explainer = {'pyexp':py_crossinter_obj, 
                         'LIME': lime_obj, 
                         'py_ctgan' : py_ctgan_obj, 
                         'py_copulagan' : py_copulagan_obj, 
                         'py_tvae' : py_tvae_obj, 
                         'py_gcopula' : py_gcopula_obj 
                         }
        
        # 'mcrossinter' : mcrossinter_obj, 

        
        # write the updated object. 
        pickle.dump(all_explainer, open(save_dir+'/all_explainer_'+row_index+'.pkl','wb'))
        ################### Loading Already Trained Files ########## 
        # '''

        if debug : 
            break

        # print('finished {}/{} commits'.format(str(i+1), str(len(feature_df))))

    # pickle.dump(mBase_time, open(save_dir+'/mBase_time'+'.pkl','wb'))
    # pickle.dump(pyExp_time, open(save_dir+'/pyExp_time'+'.pkl','wb'))
    # pickle.dump(lime_time, open(save_dir+'/lime_time'+'.pkl','wb'))

    # mBase.logBestParams()

def create_mExplainer(proj_name, global_model_name, x_train, x_test, y_train, y_test, df_indices, debug=False):
    
    save_dir = os.path.join(exp_dir,proj_name,global_model_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    global_model = pickle.load(open(proj_name+'_'+global_model_name+'_global_model.pkl','rb'))

    indep = x_test.columns
    dep = 'defect'
    class_label = ['clean', 'defect']
    

    # for my testing 
    mBase = MohitBase(x_train, y_train, indep, dep, global_model, class_label)


    feature_df = x_test.loc[df_indices]
    test_label = y_test.loc[df_indices]

    mBase_time = []
    pyExp_time = []
    lime_time= []
    
    def _prepareExplainerObj(usePreTrainedData=False, preTrainedData=None, search_function='ctgan',modelType='nc') : 
        if debug : 
            print("\t..starting Mohit-base with search function = ", search_function)

        start = time.time()   
        mBase_obj = mBase.explain(X_explain,
                                   y_explain,
                                   modelType = modelType, 
                                   cv = 1,
                                   search_function = search_function,
                                   usePreTrainedData = usePreTrainedData,
                                   preTrainedData = preTrainedData,
                                   debug=debug)
        end = time.time()
        elapsed = end - start
        
        if debug : 
            print("\t..done Mohit-base with search function = ", search_function)

        # add row index to the object 
        mBase_obj['commit_id'] = row_index
        # add time stamp to the object
        mBase_obj['time'] = elapsed

        return mBase_obj

    for i in tqdm(range(0,len(feature_df))):
        X_explain = feature_df.iloc[[i]]
        y_explain = test_label.iloc[[i]]

        row_index = str(X_explain.index[0])

        if debug : 
            print("\nFor Row Index", row_index)
        
        preTrainedAllData = pickle.load(open(os.path.join(d_dir,proj_name,global_model_name,'syndata_'+row_index+'.pkl'),'rb'))

        # #################################
        mctgan_obj = _prepareExplainerObj(usePreTrainedData=True, preTrainedData=preTrainedAllData['ctgan'],search_function='ctgan',modelType='nr') 
        mcopulagan_obj = _prepareExplainerObj(usePreTrainedData=True, preTrainedData=preTrainedAllData['copulagan'],search_function='copulagan',modelType='nr')
        mtvae_obj = _prepareExplainerObj(usePreTrainedData=True, preTrainedData=preTrainedAllData['tvae'],search_function='tvae',modelType='nr')
        mgcopula_obj = _prepareExplainerObj(usePreTrainedData=True, preTrainedData=preTrainedAllData['gcopula'],search_function='gcopula',modelType='nr')
        mcrossinter_obj = _prepareExplainerObj(usePreTrainedData=True, preTrainedData=preTrainedAllData['crossoverinterpolation'],search_function='crossoverinterpolation',modelType='nr')

        ################### Loading Already Trained Files ########## 
        # load already trained file, from previous executions. 
        # all_explainer = pickle.load(open(save_dir+'/all_explainer_mnc'+row_index+'.pkl','rb'))

        # update the object
        all_explainer = {
                         'mctgan' : mctgan_obj, 
                         'mcopulagan' : mcopulagan_obj, 
                         'mtvae' : mtvae_obj, 
                         'mgcopula' : mgcopula_obj, 
                         'mcrossinter' : mcrossinter_obj, 
                        }
        
        # write the updated object. 
        pickle.dump(all_explainer, open(save_dir+'/all_explainer_mnr'+row_index+'.pkl','wb'))
        # pickle.dump(all_explainer, open(save_dir+'/all_explainer_mnc'+row_index+'.pkl','wb'))
        ################### Loading Already Trained Files ########## 
        # '''

        if debug : 
            break

        # print('finished {}/{} commits'.format(str(i+1), str(len(feature_df))))

    # pickle.dump(mBase_time, open(save_dir+'/mBase_time'+'.pkl','wb'))
    # pickle.dump(pyExp_time, open(save_dir+'/pyExp_time'+'.pkl','wb'))
    # pickle.dump(lime_time, open(save_dir+'/lime_time'+'.pkl','wb'))

    # mBase.logBestParams()


def train_explainer(proj_name, global_model_name,debug=False):
    x_train, x_test, y_train, y_test = prepare_data(proj_name, mode = 'all')

    correctly_predict_indice = get_correctly_predicted_defective_commit_indices(proj_name, global_model_name, x_test, y_test)
    correctly_predict_indice = set(correctly_predict_indice)
    # create_explainer(proj_name, global_model_name, x_train, x_test, y_train, y_test, correctly_predict_indice,debug=debug)
    # create_baseExplainer(proj_name, global_model_name, x_train, x_test, y_train, y_test, correctly_predict_indice,debug=debug)
    create_mExplainer(proj_name, global_model_name, x_train, x_test, y_train, y_test, correctly_predict_indice,debug=debug)


def _create_synthetic_data(proj_name, global_model_name, x_train, x_test, y_train, y_test, df_indices, synthesizer_type=[], debug=False):
    
    save_dir = os.path.join(d_dir,proj_name,global_model_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    global_model = pickle.load(open(proj_name+'_'+global_model_name+'_global_model.pkl','rb'))

    indep = x_test.columns
    dep = 'defect'
    class_label = ['clean', 'defect']

    # Create Mohit Base Object which will be invoked again and again to create dataset. 
    mBase = MohitBase(x_train, y_train, indep, dep, global_model, class_label)

    feature_df = x_test.loc[df_indices]
    test_label = y_test.loc[df_indices]

    time_dict = {}
    
    for i in tqdm(range(0,len(feature_df))):
        X_explain = feature_df.iloc[[i]]
        y_explain = test_label.iloc[[i]]

        row_index = str(X_explain.index[0])
        combined_data_obj = {}
        combined_data_obj['commit_id'] = row_index # add row index to the object

        if debug : 
            print("\nFor Row Index", row_index)

        if len(synthesizer_type) == 0 or 'real' in synthesizer_type : 
            ########## Extracting Real Neighbourhood ############
            if debug : 
                print("\tObtaining Real data in the Neighbourhood for the Sample under examination")
            readData = mBase.provideNeighbourhood(X_explain, y_explain, debug=debug)
            combined_data_obj['real'] = readData
            if debug : 
                print("\t..Real Neighbourhood extracted")
            ########## Extracting Real Neighbourhood ############


        if len(synthesizer_type) == 0 or 'ctgan' in synthesizer_type : 
            ########## Generating Data Using CTGan ############
            if debug : 
                print("\tstarting Mohit-base CTGan")
            if i == 0 : 
                time_dict['ctgan'] = []

            start = time.time()
            sampledData = mBase.generate_instance_sdv(X_explain, y_explain, 'ctgan' , debug=debug)
            end = time.time()
            elapsed = end - start
            sampledData['time'] = elapsed
            time_dict['ctgan'].append(elapsed)
            combined_data_obj['ctgan'] = sampledData

            if debug : 
                print("\t..done Mohit-base CTGan")
                
            if len(synthesizer_type) == 1 and debug : 
                break
            ########## Generating Data Using CTGan ############

        

        if len(synthesizer_type) == 0 or 'copulagan' in synthesizer_type : 
            ########## Generating Data Using copulagan ############
            if debug : 
                print("\tstarting Mohit-base copulagan")
            if i == 0 : 
                time_dict['copulagan'] = []

            start = time.time()
            sampledData = mBase.generate_instance_sdv(X_explain, y_explain, 'copulagan' , debug=debug)
            end = time.time()
            elapsed = end - start
            sampledData['time'] = elapsed
            time_dict['copulagan'].append(elapsed)
            combined_data_obj['copulagan'] = sampledData

            if debug : 
                print("\t..done Mohit-base copulagan")
            ########## Generating Data Using copulagan ############
        
        if len(synthesizer_type) == 0 or 'tvae' in synthesizer_type : 
            ########## Generating Data Using TVAE ############
            if debug : 
                print("\tstarting Mohit-base tvae")
            if i == 0 : 
                time_dict['tvae'] = []

            start = time.time()
            sampledData = mBase.generate_instance_sdv(X_explain, y_explain, 'tvae' , debug=debug)
            end = time.time()
            elapsed = end - start
            sampledData['time'] = elapsed
            time_dict['tvae'].append(elapsed)
            combined_data_obj['tvae'] = sampledData

            if debug : 
                print("\t..done Mohit-base tvae")
            ########## Generating Data Using TVAE ############
        
        if len(synthesizer_type) == 0 or 'gcopula' in synthesizer_type : 
            ########## Generating Data Using gcopula ############
            if debug : 
                print("\tstarting Mohit-base gcopula")
            if i == 0 : 
                time_dict['gcopula'] = []

            start = time.time()
            sampledData = mBase.generate_instance_sdv(X_explain, y_explain, 'gcopula' , debug=debug)
            end = time.time()
            elapsed = end - start
            sampledData['time'] = elapsed
            time_dict['gcopula'].append(elapsed)
            combined_data_obj['gcopula'] = sampledData

            if debug : 
                print("\t..done Mohit-base gcopula")
            ########## Generating Data Using gcopula ############

        if len(synthesizer_type) == 0 or 'crossoverinterpolation' in synthesizer_type : 
            ########## Generating Data Using crossover_interpolation ############
            if debug : 
                print("\tstarting crossoverinterpolation")
            if i == 0 : 
                time_dict['crossoverinterpolation'] = []

            start = time.time()
            sampledData = mBase.generate_instance_crossover_interpolation(X_explain, y_explain, debug=debug)
            end = time.time()
            elapsed = end - start
            sampledData['time'] = elapsed
            time_dict['crossoverinterpolation'].append(elapsed)
            combined_data_obj['crossoverinterpolation'] = sampledData

            if debug : 
                print("\t..done crossoverinterpolation")
            ########## Generating Data Using crossover_interpolation ############

        if len(synthesizer_type) == 0 or 'randomperturbation' in synthesizer_type : 
            ########## Generating Data Using random_perturbation ############
            if debug : 
                print("\tstarting randomperturbation")
            if i == 0 : 
                time_dict['randomperturbation'] = []

            start = time.time()
            sampledData = mBase.generate_instance_random_perturbation(X_explain,debug=debug)
            end = time.time()
            elapsed = end - start
            sampledData['time'] = elapsed
            time_dict['randomperturbation'].append(elapsed)
            combined_data_obj['lime'] = sampledData

            if debug : 
                print("\t..done randomperturbation")
            ########## Generating Data Using random_perturbation ############
        
        # ########## using old trained data ###############
        # old_data = pickle.load(open(save_dir+'/syndata_'+row_index+'.pkl','rb'))
        # combined_data_obj['lime'] = old_data['lime']
        # combined_data_obj['pyexp'] = old_data['pyexp']
        # try : 
        #     combined_data_obj['ctgan'] = old_data['mbase']
        # except : 
        #     combined_data_obj['ctgan'] = old_data['ctgan']
        # ############ using old trained data ###############


        pickle.dump(combined_data_obj, open(save_dir+'/syndata_'+row_index+'.pkl','wb'))

        if debug : 
            break

    pickle.dump(time_dict, open(save_dir+'/time_dict'+'.pkl','wb'))

    # mBase.logBestParams()

def create_synthetic_data(proj_name, global_model_name, synthesizer_type=[], debug=False):
    x_train, x_test, y_train, y_test = prepare_data(proj_name, mode = 'all')

    correctly_predict_indice = get_correctly_predicted_defective_commit_indices(proj_name, global_model_name, x_test, y_test)
    correctly_predict_indice = set(correctly_predict_indice)
    _create_synthetic_data(proj_name, global_model_name, x_train, x_test, y_train, y_test, correctly_predict_indice, synthesizer_type, debug=debug)

proj_name = sys.argv[1]
proj_name = proj_name.lower()
global_model = sys.argv[2]
global_model = global_model.upper()
debug = (sys.argv[3] == '1' ) # value 1 means, enabled in debug mode.

if debug : 
    print("launched in debug mode")


if proj_name not in ['openstack','qt'] or global_model not in ['RF','LR']:
    print('project name must be "openstack" or "qt".')
    print('global model name must be "RF" or "LR".')
    
else:
    print(proj_name, global_model)
    print('training global model')
    train_global_model_runner(proj_name, global_model)
    print('\tfinished training global model')
    

    # ########### For Just Creating Synthetic Data ############
    # print('creating custom datasets')
    # # create_synthetic_data(proj_name, global_model,synthesizer_type = [] , debug=debug)
    # create_synthetic_data(proj_name, global_model,synthesizer_type = [] , debug=debug)
    # print('\tfinished creation of custom datasets')


    ########### For Creation of Explainers ###############
    print('training explainers')
    train_explainer(proj_name, global_model,debug=debug)
    print('\tfinished training explainers')


