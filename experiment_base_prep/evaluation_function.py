from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import auc, roc_auc_score, f1_score, confusion_matrix, classification_report, recall_score
import pandas as pd
import numpy as np
import seaborn as sns

from my_util import *
from lime.lime.lime_tabular import LimeTabularExplainer

import matplotlib.pyplot as plt

import os, pickle, time, re, sys, operator
from datetime import datetime
from collections import Counter

sys.path.append(os.path.abspath('../'))
from pyexplainer.pyexplainer_pyexplainer import *
from tqdm import tqdm 

data_path = './dataset/'
result_dir = './eval_result/'
dump_dataframe_dir = './prediction_result/'
exp_dir = './explainer_object/'
d_dir ='./synthetic_data/'

fig_dir = result_dir+'figures/'

flip_sign_dict = {
    '<': '>=',
    '>': '<=',
    '=': '!=',
    '>=': '<',
    '<=': '>',
    '!=': '=='
}

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)


####### Helper Functions #####
###############################################################################
def aggregate_list(l):
    return np.mean(l), np.median(l)

def prepare_data_for_testing(proj_name, global_model_name = 'RF'):
    global_model_name = global_model_name.upper()
    global_model = pickle.load(open(proj_name+'_'+global_model_name+'_global_model.pkl','rb'))

    correctly_predict_df = pd.read_csv(dump_dataframe_dir+proj_name+'_'+global_model_name+'_correctly_predict_as_defective.csv')
    correctly_predict_df = correctly_predict_df.set_index('commit_id')

    dep = 'defect'
    indep = correctly_predict_df.columns[:-3] # exclude the last 3 columns

    feature_df = correctly_predict_df.loc[:, indep]
    
    return global_model, correctly_predict_df, indep, dep, feature_df
    
def get_prediction_result_df(proj_name, global_model_name):
    global_model_name = global_model_name.upper()
    if global_model_name not in ['RF','LR']:
        print('wrong global model name. the global model name must be RF or LR')
        return
    
    prediction_df_dir = dump_dataframe_dir+proj_name+'_'+global_model_name+'_prediction_result.csv'
    prediction_df = pd.read_csv(prediction_df_dir)
    prediction_df = prediction_df.set_index('commit_id')
    
    return prediction_df

def get_recall_at_k_percent_effort(percent_effort, result_df_arg, real_buggy_commits):
    cum_LOC_k_percent = (percent_effort/100)*result_df_arg.iloc[-1]['cum_LOC']
    buggy_line_k_percent =  result_df_arg[result_df_arg['cum_LOC'] <= cum_LOC_k_percent]
    buggy_commit = buggy_line_k_percent[buggy_line_k_percent['defect']==True]
    recall_k_percent_effort = len(buggy_commit)/float(len(real_buggy_commits))
    
    return recall_k_percent_effort

def eval_global_model(proj_name, prediction_df):
    ## since ld metric in openstack is removed by using autospearman, so this code is needed
    ## but this is not problem for qt
    
    if proj_name == 'openstack':
        x_train_original, x_test_original = prepare_data_all_metrics(proj_name, mode='all')
        prediction_df = prediction_df.copy()
        prediction_df['ld'] = list(x_test_original['ld'])
        
    prediction_df = prediction_df[['la','ld', 'pred', 'defective_prob' ,'defect']]
    prediction_df['LOC'] = prediction_df['la']+prediction_df['ld']
    
    prediction_df['defect_density'] = prediction_df['defective_prob']/prediction_df['LOC']
    prediction_df['actual_defect_density'] = prediction_df['defect']/prediction_df['LOC'] #defect density
    
    prediction_df = prediction_df.fillna(0)
    prediction_df = prediction_df.replace(np.inf, 0)
    
    prediction_df = prediction_df.sort_values(by='defect_density',ascending=False)
    
    actual_result_df = prediction_df.sort_values(by='actual_defect_density',ascending=False)
    actual_worst_result_df = prediction_df.sort_values(by='actual_defect_density',ascending=True)

    prediction_df['cum_LOC'] = prediction_df['LOC'].cumsum()
    actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
    actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()

    real_buggy_commits = prediction_df[prediction_df['defect'] == True]
    
    
    AUC = roc_auc_score(prediction_df['defect'], prediction_df['defective_prob'])
    f1 = f1_score(prediction_df['defect'], prediction_df['pred'])
    
    ifa = real_buggy_commits.iloc[0]['cum_LOC']

    cum_LOC_20_percent = 0.2*prediction_df.iloc[-1]['cum_LOC']
    buggy_line_20_percent = prediction_df[prediction_df['cum_LOC'] <= cum_LOC_20_percent]
    buggy_commit = buggy_line_20_percent[buggy_line_20_percent['defect']==True]
    recall_20_percent_effort = len(buggy_commit)/float(len(real_buggy_commits))
    
    # find P_opt
    percent_effort_list = []
    predicted_recall_at_percent_effort_list = []
    actual_recall_at_percent_effort_list = []
    actual_worst_recall_at_percent_effort_list = []
    
    for percent_effort in np.arange(10,101,10):
        predicted_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, prediction_df, real_buggy_commits)
        actual_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_result_df, real_buggy_commits)
        actual_worst_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_worst_result_df, real_buggy_commits)
        
        percent_effort_list.append(percent_effort/100)
        
        predicted_recall_at_percent_effort_list.append(predicted_recall_k_percent_effort)
        actual_recall_at_percent_effort_list.append(actual_recall_k_percent_effort)
        actual_worst_recall_at_percent_effort_list.append(actual_worst_recall_k_percent_effort)

    p_opt = 1 - ((auc(percent_effort_list, actual_recall_at_percent_effort_list) - 
                 auc(percent_effort_list, predicted_recall_at_percent_effort_list)) /
                (auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                auc(percent_effort_list, actual_worst_recall_at_percent_effort_list)))

    print('AUC: {}, F1: {}, IFA: {}, Recall@20%Effort: {}, Popt: {}'.format(AUC,f1,ifa,recall_20_percent_effort,p_opt))
    print(classification_report(prediction_df['defect'], prediction_df['pred']))

def get_global_model_evaluation_result(proj_name):
    print('RF global model result')
    rf_prediction_df = get_prediction_result_df(proj_name, 'rf')
    eval_global_model(proj_name, rf_prediction_df)

    print('-'*100)
    
    print('LR global model result')
    lr_prediction_df = get_prediction_result_df(proj_name, 'lr')
    eval_global_model(proj_name, lr_prediction_df)


####### RQ1 Evaluation #####
###############################################################################
def rq1_preprocess(proj_name, global_model_name, debug=False):
    global_model, correctly_predict_df, indep, dep, feature_df = prepare_data_for_testing(proj_name, global_model_name)
    all_eval_result = pd.DataFrame()
    
    def prepareSeries(data_dict, data_model, lime_similarityval, pyexp_similarityval) : 
        synthetic_data = data_dict['synthetic_data'].loc[:, indep].values
        euc_dist = euclidean_distances(X_explain.values, synthetic_data)
        dist_mean, dist_med = aggregate_list(euc_dist)
        similarity = 1 / dist_mean
        lime_score = 0 if data_model == 'lime' else  (similarity/lime_similarityval - 1)*100
        pyexp_score = 0 if (data_model == 'lime' or data_model =='pyexp') else (similarity/pyexp_similarityval - 1)*100
        serie = pd.Series(data=[proj_name, row_index, data_model, dist_med, similarity, lime_score, pyexp_score])
        return serie

    for i in tqdm(range(0,len(feature_df))):

        X_explain = feature_df.iloc[[i]]
        row_index = str(X_explain.index[0])
    
        data_obj = pickle.load(open(os.path.join(d_dir,proj_name,global_model_name,'syndata_'+row_index+'.pkl'),'rb'))
       
        lime_serie =  prepareSeries(data_obj['lime'],'lime',0,0)
        lime_similarityval = lime_serie[4]
        pyexp_serie = prepareSeries(data_obj['crossoverinterpolation'],'pyexp',lime_similarityval,0)
        pyexp_similarityval = pyexp_serie[4]

        all_eval_result = all_eval_result.append( lime_serie ,ignore_index=True)
        all_eval_result = all_eval_result.append( pyexp_serie ,ignore_index=True)
        all_eval_result = all_eval_result.append( prepareSeries(data_obj['ctgan'],'ctgan', lime_similarityval, pyexp_similarityval) ,ignore_index=True)
        all_eval_result = all_eval_result.append( prepareSeries(data_obj['tvae'],'tvae', lime_similarityval, pyexp_similarityval) ,ignore_index=True)
        all_eval_result = all_eval_result.append( prepareSeries(data_obj['gcopula'],'gcopula', lime_similarityval, pyexp_similarityval) ,ignore_index=True)
        all_eval_result = all_eval_result.append( prepareSeries(data_obj['copulagan'],'copulagan',lime_similarityval, pyexp_similarityval) ,ignore_index=True)

        if debug : 
            break
        
    all_eval_result.columns =['project', 'commit id', 'method', 'euc_dist_med','similarity','lime_similarity_score', 'pyexp_similarity_score']
    all_eval_result.to_csv(result_dir+'RQ1_syndata_'+proj_name+'_'+global_model_name+'.csv',index=False)
    print('finished RQ1 Preprocess of',proj_name,', globla model is',global_model_name)

def show_rq1_images() :
    openstack_rf = pd.read_csv(result_dir+'RQ1_syndata_openstack_RF.csv')
    qt_rf = pd.read_csv(result_dir+'RQ1_syndata_qt_RF.csv')
    result_rf = pd.concat([openstack_rf, qt_rf])
    result_rf['global_model'] = 'RF'
    
    openstack_lr = pd.read_csv(result_dir+'RQ1_syndata_openstack_LR.csv')
    qt_lr = pd.read_csv(result_dir+'RQ1_syndata_qt_LR.csv')
    result_lr = pd.concat([openstack_lr, qt_lr])
    result_lr['global_model'] = 'LR'
    
    all_result = pd.concat([result_rf, result_lr])

    fig, axs = plt.subplots(1,2, figsize=(10,6))

    axs[0].set_title('RF')
    axs[1].set_title('LR')
    
    axs[0].set(ylim=(0, 5000))
    axs[1].set(ylim=(0, 5000))
    
    sns.boxplot(data=result_rf, x='project', y='euc_dist_med', hue='method', ax=axs[0])
    sns.boxplot(data=result_lr, x='project', y='euc_dist_med', hue='method', ax=axs[1])
    
    plt.show()
    
    all_result.to_csv(result_dir+'/RQ1_syndata.csv',index=False)
    fig.savefig(fig_dir+'RQ1.png')

def show_rq1_scores(): 
    openstack_rf = pd.read_csv(result_dir+'RQ1_syndata_openstack_RF.csv')
    qt_rf = pd.read_csv(result_dir+'RQ1_syndata_qt_RF.csv')
    result_rf = pd.concat([openstack_rf, qt_rf])
    result_rf['global_model'] = 'RF'
    
    openstack_lr = pd.read_csv(result_dir+'RQ1_syndata_openstack_LR.csv')
    qt_lr = pd.read_csv(result_dir+'RQ1_syndata_qt_LR.csv')
    result_lr = pd.concat([openstack_lr, qt_lr])
    result_lr['global_model'] = 'LR'
    
    all_result = pd.concat([result_rf, result_lr])
    
    # group the data based on the 'method' feature and find the min/max values of features 5 and 6 for each group
    grouped_data = all_result.groupby('method').agg({'similarity':['mean'],'lime_similarity_score': ['min', 'mean', 'max'], 'pyexp_similarity_score': ['min','mean', 'max']})

    # print the resulting grouped data
    print(grouped_data) 


####### RQ2 Evaluation #####
###############################################################################
def rq2_preprocess(proj_name, global_model_name, debug=False):
    global_model, correctly_predict_df, indep, dep, feature_df = prepare_data_for_testing(proj_name, global_model_name)
    time_result = pd.DataFrame()

    def prepareTimeSeries(data_dict, data_model) : 
        time = data_dict['time']
        serie = pd.Series(data=[proj_name, row_index, data_model, time])
        return serie

    for i in tqdm(range(0,len(feature_df))):

        X_explain = feature_df.iloc[[i]]
        row_index = str(X_explain.index[0])
    
        data_obj = pickle.load(open(os.path.join(d_dir,proj_name,global_model_name,'syndata_'+row_index+'.pkl'),'rb'))

        time_result = time_result.append( prepareTimeSeries(data_obj['lime'],'lime') ,ignore_index=True)
        time_result = time_result.append( prepareTimeSeries(data_obj['crossoverinterpolation'],'pyexp') ,ignore_index=True)
        time_result = time_result.append( prepareTimeSeries(data_obj['ctgan'],'ctgan') ,ignore_index=True)
        time_result = time_result.append( prepareTimeSeries(data_obj['tvae'],'tvae') ,ignore_index=True)
        time_result = time_result.append( prepareTimeSeries(data_obj['gcopula'],'gcopula') ,ignore_index=True)
        time_result = time_result.append( prepareTimeSeries(data_obj['copulagan'],'copulagan') ,ignore_index=True)

        if debug : 
            break

    time_result.columns =['project', 'commit id', 'method', 'time']
    
    time_result.to_csv(result_dir+'RQ2_syndata_time_'+proj_name+'_'+global_model_name+'.csv',index=False)
    print('finished RQ2 Preprocess of',proj_name,', globla model is',global_model_name)

def show_rq2_images() :
    openstack_rf = pd.read_csv(result_dir+'RQ2_syndata_time_openstack_RF.csv')
    qt_rf = pd.read_csv(result_dir+'RQ2_syndata_time_qt_RF.csv')
    result_rf = pd.concat([openstack_rf, qt_rf])
    result_rf['global_model'] = 'RF'
    
    openstack_lr = pd.read_csv(result_dir+'RQ2_syndata_time_openstack_LR.csv')
    qt_lr = pd.read_csv(result_dir+'RQ2_syndata_time_qt_LR.csv')
    result_lr = pd.concat([openstack_lr, qt_lr])
    result_lr['global_model'] = 'LR'
    
    all_result = pd.concat([result_rf, result_lr])

    fig, axs = plt.subplots(1,2, figsize=(10,6))

    axs[0].set_title('RF')
    axs[1].set_title('LR')
    
    axs[0].set(ylim=(0, 300))
    axs[1].set(ylim=(0, 300))
    
    sns.boxplot(data=result_rf, x='project', y='time', hue='method', ax=axs[0])
    sns.boxplot(data=result_lr, x='project', y='time', hue='method', ax=axs[1])
    
    plt.show()
    
    all_result.to_csv(result_dir+'/RQ2_syndata.csv',index=False)
    fig.savefig(fig_dir+'RQ2.png')

def show_rq2_scores(): 
    openstack_rf = pd.read_csv(result_dir+'RQ2_syndata_time_openstack_RF.csv')
    qt_rf = pd.read_csv(result_dir+'RQ2_syndata_time_qt_RF.csv')
    result_rf = pd.concat([openstack_rf, qt_rf])
    result_rf['global_model'] = 'RF'
    
    openstack_lr = pd.read_csv(result_dir+'RQ2_syndata_time_openstack_LR.csv')
    qt_lr = pd.read_csv(result_dir+'RQ2_syndata_time_qt_LR.csv')
    result_lr = pd.concat([openstack_lr, qt_lr])
    result_lr['global_model'] = 'LR'
    
    all_result = pd.concat([result_rf, result_lr])
    
    # group the data based on the 'method' feature and find the min/max values of features 5 and 6 for each group
    grouped_data = all_result.groupby('method').agg({'time': ['min','mean']})

    print("gcopula is faster than pyexp by ",grouped_data.loc['pyexp', ('time', 'mean')]/grouped_data.loc['gcopula', ('time', 'mean')], " factors.")
    print()

    # print the resulting grouped data
    print(grouped_data) 


###############################################################################
def test_file_sync():
    print("Version 6.3.{}".format(datetime.now().strftime("%H:%M:%S")))


####### RQ3 Evaluation #####
###############################################################################
def rq3_showsampledata(proj_name, global_model_name, debug=False):
    global_model, correctly_predict_df, indep, dep, feature_df = prepare_data_for_testing(proj_name, global_model_name)
    
    print(feature_df.head())
    summary = feature_df.describe()
    print(summary)

def rq3_preprocess(proj_name, global_model_name, debug=False):
    global_model, correctly_predict_df, indep, dep, feature_df = prepare_data_for_testing(proj_name, global_model_name)
    all_eval_result = pd.DataFrame()
    
    def processDuplicates(df ) : 
       df_rounded = df.round(1)
       duplicates = df_rounded.duplicated()
       num_duplicates = duplicates.sum()
       return num_duplicates

    def prepareSeries(data_dict, data_model) : 
        synthetic_data = data_dict['synthetic_data'].loc[:, indep].values
        all_count = len(synthetic_data)
        dup_count = processDuplicates(synthetic_data)

        serie = pd.Series(data=[proj_name, row_index, data_model, dup_count/all_count*100])
        return serie

    for i in tqdm(range(0,len(feature_df))):

        X_explain = feature_df.iloc[[i]]
        row_index = str(X_explain.index[0])
    
        data_obj = pickle.load(open(os.path.join(d_dir,proj_name,global_model_name,'syndata_'+row_index+'.pkl'),'rb'))

        all_eval_result = all_eval_result.append( prepareSeries(data_obj['lime']),ignore_index=True)
        all_eval_result = all_eval_result.append( prepareSeries(data_obj['crossoverinterpolation'],'pyexp') ,ignore_index=True)
        all_eval_result = all_eval_result.append( prepareSeries(data_obj['ctgan'],'ctgan') ,ignore_index=True)
        all_eval_result = all_eval_result.append( prepareSeries(data_obj['tvae'],'tvae') , ignore_index=True)
        all_eval_result = all_eval_result.append( prepareSeries(data_obj['gcopula'],'gcopula') ,ignore_index=True)
        all_eval_result = all_eval_result.append( prepareSeries(data_obj['copulagan'],'copulagan') ,ignore_index=True)

        if debug : 
            break
        
    all_eval_result.columns =['project', 'commit id', 'method', 'percent_duplicates']
    all_eval_result.to_csv(result_dir+'RQ3_syndata_'+proj_name+'_'+global_model_name+'.csv',index=False)
    print('finished RQ1 Preprocess of',proj_name,', globla model is',global_model_name)

def show_rq3_images() :
    openstack_rf = pd.read_csv(result_dir+'RQ2_syndata_time_openstack_RF.csv')
    qt_rf = pd.read_csv(result_dir+'RQ2_syndata_time_qt_RF.csv')
    result_rf = pd.concat([openstack_rf, qt_rf])
    result_rf['global_model'] = 'RF'
    
    openstack_lr = pd.read_csv(result_dir+'RQ2_syndata_time_openstack_LR.csv')
    qt_lr = pd.read_csv(result_dir+'RQ2_syndata_time_qt_LR.csv')
    result_lr = pd.concat([openstack_lr, qt_lr])
    result_lr['global_model'] = 'LR'
    
    all_result = pd.concat([result_rf, result_lr])

    fig, axs = plt.subplots(1,2, figsize=(10,6))

    axs[0].set_title('RF')
    axs[1].set_title('LR')
    
    axs[0].set(ylim=(0, 300))
    axs[1].set(ylim=(0, 300))
    
    sns.boxplot(data=result_rf, x='project', y='time', hue='method', ax=axs[0])
    sns.boxplot(data=result_lr, x='project', y='time', hue='method', ax=axs[1])
    
    plt.show()
    
    all_result.to_csv(result_dir+'/RQ2_syndata.csv',index=False)
    fig.savefig(fig_dir+'RQ2.png')

def show_rq3_scores(): 
    openstack_rf = pd.read_csv(result_dir+'RQ2_syndata_time_openstack_RF.csv')
    qt_rf = pd.read_csv(result_dir+'RQ2_syndata_time_qt_RF.csv')
    result_rf = pd.concat([openstack_rf, qt_rf])
    result_rf['global_model'] = 'RF'
    
    openstack_lr = pd.read_csv(result_dir+'RQ2_syndata_time_openstack_LR.csv')
    qt_lr = pd.read_csv(result_dir+'RQ2_syndata_time_qt_LR.csv')
    result_lr = pd.concat([openstack_lr, qt_lr])
    result_lr['global_model'] = 'LR'
    
    all_result = pd.concat([result_rf, result_lr])
    
    # group the data based on the 'method' feature and find the min/max values of features 5 and 6 for each group
    grouped_data = all_result.groupby('method').agg({'time': ['min','mean']})

    print("gcopula is faster than pyexp by ",grouped_data.loc['pyexp', ('time', 'mean')]/grouped_data.loc['gcopula', ('time', 'mean')], " factors.")
    print()

    # print the resulting grouped data
    print(grouped_data) 



####### RQ4 Evaluation #####
############################################################################### 
list_of_models = ['lime' , 'pyexp', 'py_ctgan', 'py_copulagan', 'py_tvae', 'py_gcopula', 'mctgan', 'mcopulagan', 'mtvae', 'mgcopula' , 'mcrossinter', 'mr_ctgan', 'mr_copulagan', 'mr_tvae', 'mr_gcopula', 'mr_crossinter' ]

def rq4_eval(proj_name, global_model_name, debug = False):
    global_model_name = global_model_name.upper()
    
    global_model, correctly_predict_df, indep, dep, feature_df = prepare_data_for_testing(proj_name, global_model_name)
    all_eval_result = pd.DataFrame()
    all_labels = {'lime' : []}
    all_probs = {'lime' : []}

    def _prepareSerie(exp, exp_name) : 
        synthetic_data = exp['synthetic_data'].values # synthetic data to work upon
        local_model = exp['local_model'] # local explainer model 

        global_pred = global_model.predict(synthetic_data) # for the model to be explained 
        local_pred = local_model.predict(synthetic_data) # predictions for the synthetic data for local model, to understand how well local model maps to global model 

        local_prob_inter = local_model.predict_proba(synthetic_data) 
        local_prob = local_prob_inter[:,1] # proba for local model 

        if  exp_name in all_labels.keys() : 
            all_labels[exp_name].extend(list(global_pred)) # store for later use
        else : 
            all_labels[exp_name] = list(global_pred) # store for later use

        if  exp_name in all_probs.keys() : 
            all_probs[exp_name].extend(list(local_prob)) # store for later use
        else :
            all_probs[exp_name] = list(local_prob) # store for later use


        exp_auc = roc_auc_score(global_pred, local_prob) # roc auc 
        exp_f1 = f1_score(global_pred, local_pred) # f1 
        
        exp_serie = pd.Series(data=[proj_name, row_index, exp_name, exp_auc, exp_f1]) # embed data in the serie and return
        return exp_serie
       
    for i in tqdm(range(0,len(feature_df))):
        if debug : 
            if i > 0 : 
                break
        X_explain = feature_df.iloc[[i]]

        row_index = str(X_explain.index[0])

        exp_obj = pickle.load(open(os.path.join(exp_dir,proj_name,global_model_name,'all_explainer_'+row_index+'.pkl'),'rb'))
        
        ######### Processing of Line happens Little different so keep it separate ##########
        lime_exp = exp_obj['LIME']
        # this data can be used with global model only
        lime_exp_synthetic_data = lime_exp['synthetic_instance_for_global_model']
        # this data can be used with local model only
        lime_exp_synthetic_data_local = lime_exp['synthetic_instance_for_lobal_model']
        # get model from lime 
        lime_exp_local_model = lime_exp['local_model']
        # get predictions and Probs 
        lime_exp_global_pred = global_model.predict(lime_exp_synthetic_data)
        lime_exp_local_prob = lime_exp_local_model.predict(lime_exp_synthetic_data_local)
        lime_exp_local_pred = np.round(lime_exp_local_prob)

        all_labels['lime'].extend(list(lime_exp_global_pred))
        all_probs['lime'].extend(list(lime_exp_local_prob))

        lime_auc = roc_auc_score(lime_exp_global_pred, lime_exp_local_prob)
        lime_f1 = f1_score(lime_exp_global_pred, lime_exp_local_pred)

        lime_exp_serie = pd.Series(data=[proj_name, row_index, 'lime', lime_auc, lime_f1])

        all_eval_result = all_eval_result.append(lime_exp_serie, ignore_index=True)
        ######### Processing of Line happens Little different so keep it separate ##########
        
        
        all_eval_result = all_eval_result.append(_prepareSerie(exp_obj['pyexp'], 'pyexp'),ignore_index=True) # for 'PyExplainer' with crossinter
        all_eval_result = all_eval_result.append(_prepareSerie(exp_obj['py_ctgan'], 'py_ctgan'),ignore_index=True) # for 'PyExplainer' with ctgan
        all_eval_result = all_eval_result.append(_prepareSerie(exp_obj['py_copulagan'], 'py_copulagan'),ignore_index=True) # for 'PyExplainer' with copulagan
        all_eval_result = all_eval_result.append(_prepareSerie(exp_obj['py_tvae'], 'py_tvae'),ignore_index=True) # for 'PyExplainer' with tvae
        all_eval_result = all_eval_result.append(_prepareSerie(exp_obj['py_gcopula'], 'py_gcopula'),ignore_index=True) # for 'PyExplainer' with gcopula

        exp_obj = pickle.load(open(os.path.join(exp_dir,proj_name,global_model_name,'all_explainer_mnc'+row_index+'.pkl'),'rb'))     

        all_eval_result = all_eval_result.append(_prepareSerie(exp_obj['mctgan'], 'mctgan'),ignore_index=True) # for 'mctgan'
        all_eval_result = all_eval_result.append(_prepareSerie(exp_obj['mcopulagan'], 'mcopulagan'),ignore_index=True) # for 'mcopulagan'
        all_eval_result = all_eval_result.append(_prepareSerie(exp_obj['mtvae'], 'mtvae'),ignore_index=True) # for 'mtvae'
        all_eval_result = all_eval_result.append(_prepareSerie(exp_obj['mgcopula'], 'mgcopula'),ignore_index=True) # for 'mgcopula'
        all_eval_result = all_eval_result.append(_prepareSerie(exp_obj['mcrossinter'], 'mcrossinter'),ignore_index=True) # for 'mcrossinter'

        exp_obj = pickle.load(open(os.path.join(exp_dir,proj_name,global_model_name,'all_explainer_mnr'+row_index+'.pkl'),'rb'))     

        all_eval_result = all_eval_result.append(_prepareSerie(exp_obj['mctgan'], 'mr_ctgan'),ignore_index=True) # for 'mctgan'
        all_eval_result = all_eval_result.append(_prepareSerie(exp_obj['mcopulagan'], 'mr_copulagan'),ignore_index=True) # for 'mcopulagan'
        all_eval_result = all_eval_result.append(_prepareSerie(exp_obj['mtvae'], 'mr_tvae'),ignore_index=True) # for 'mtvae'
        all_eval_result = all_eval_result.append(_prepareSerie(exp_obj['mgcopula'], 'mr_gcopula'),ignore_index=True) # for 'mgcopula'
        all_eval_result = all_eval_result.append(_prepareSerie(exp_obj['mcrossinter'], 'mr_crossinter'),ignore_index=True) # for 'mcrossinter'

    
    pred_df = pd.DataFrame()

    techs = []
    for key in all_labels.keys() : 
        techs += [key]*len(all_labels[key])
    pred_df['technique'] = techs

    labs = []
    for key in all_labels.keys() : 
        labs += all_labels[key]
    pred_df['label']  = labs

    probs = []
    for key in all_probs.keys() : 
        probs += all_probs[key]
    pred_df['prob'] = probs

    pred_df['project'] = proj_name
    
    all_eval_result.columns = ['project', 'commit id', 'method', 'AUC', 'F1']

    all_eval_result.to_csv(result_dir+'RQ4_'+proj_name+'_'+global_model_name+'_global_vs_local_synt_pred.csv',index=False)
    pred_df.to_csv(result_dir+'RQ4_'+proj_name+'_'+global_model_name+'_probability_distribution.csv',index=False)
    print('finished RQ4 of',proj_name)
    
def show_rq4_eval_result():
    openstack_rf = pd.read_csv(result_dir+'RQ4_openstack_RF_global_vs_local_synt_pred.csv')
    qt_rf = pd.read_csv(result_dir+'RQ4_qt_RF_global_vs_local_synt_pred.csv')
    result_rf = pd.concat([openstack_rf, qt_rf])
    result_rf['global_model'] = 'RF'
    
    openstack_lr = pd.read_csv(result_dir+'/RQ4_openstack_LR_global_vs_local_synt_pred.csv')
    qt_lr = pd.read_csv(result_dir+'/RQ4_qt_LR_global_vs_local_synt_pred.csv')
    result_lr = pd.concat([openstack_lr, qt_lr])
    result_lr['global_model'] = 'LR'
    
    all_result = pd.concat([result_rf, result_lr])
    
    openstack_result = all_result[all_result['project']=='openstack']
    qt_result = all_result[all_result['project']=='qt']

    fig, axs = plt.subplots(2,2, figsize=(20,10))

    axs[0,0].set_title('Openstack')
    axs[0,1].set_title('Qt')
    
    axs[0,0].set_ylim([0, 1])
    axs[0,1].set_ylim([0, 1]) 
    axs[1,0].set_ylim([0, 1])
    axs[1,1].set_ylim([0, 1])

    cols = ['Openstack','Qt']
    pad = 5 # in points

    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
        
    sns.boxplot(data=openstack_result, x='global_model', y='AUC', hue='method', ax=axs[0,0])
    sns.boxplot(data=openstack_result, x='global_model', y='F1', hue='method', ax=axs[1,0])
    sns.boxplot(data=qt_result, x='global_model', y='AUC', hue='method', ax=axs[0,1])
    sns.boxplot(data=qt_result, x='global_model', y='F1', hue='method', ax=axs[1,1])

    plt.show()
    all_result.to_csv(result_dir+'RQ4_prediction.csv',index=False)
    fig.savefig(fig_dir+'RQ4_prediction.png')

def show_rq4_prob_distribution():
    
    d = {True: 'DEFECT', False: 'CLEAN'}

    openstack_rf = pd.read_csv(result_dir+'RQ4_openstack_RF_probability_distribution.csv')
    qt_rf = pd.read_csv(result_dir+'RQ4_qt_RF_probability_distribution.csv')
    
    mask = openstack_rf.applymap(type) != bool
    openstack_rf = openstack_rf.where(mask, openstack_rf.replace(d))
    qt_rf = qt_rf.where(mask, qt_rf.replace(d))
    
    result_rf = pd.concat([openstack_rf, qt_rf])
    result_rf['global_model'] = 'RF'
    
    openstack_lr = pd.read_csv(result_dir+'RQ4_openstack_LR_probability_distribution.csv')
    qt_lr = pd.read_csv(result_dir+'RQ4_qt_LR_probability_distribution.csv')
    
    openstack_lr = openstack_lr.where(mask, openstack_lr.replace(d))
    qt_lr = qt_lr.where(mask, qt_lr.replace(d))
    
    result_lr = pd.concat([openstack_lr, qt_lr])
    result_lr['global_model'] = 'LR'
    
    all_result = pd.concat([result_rf, result_lr])
    
    
    fig, axs = plt.subplots(len(list_of_models),2, figsize=(10,10))

    for index in range(0,len(list_of_models)) : 
        axs[index,0].set_ylim([0, 1])
        axs[index,1].set_ylim([0, 1]) 

        op_data = all_result[(all_result['project']=='openstack') & (all_result['technique']==list_of_models[index])]
        qt_data = all_result[(all_result['project']=='qt') & (all_result['technique']==list_of_models[index])]
        
        sns.boxplot(data=op_data, x='global_model', y='prob', hue='label' , ax=axs[index,0])
        sns.boxplot(data=qt_data,  x='global_model', y='prob', hue='label' , ax=axs[index,1])

        axs[index,0].axhline(0.5, ls='--')
        axs[index,1].axhline(0.5, ls='--')
    
    rows = list_of_models
    cols = ['Openstack','Qt']
    plt.setp(axs.flat, xlabel='Technique', ylabel='Probability')
    pad = 5 # in points

    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axs[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    
    plt.show()
    all_result.to_csv(result_dir+'RQ4_prediction_prob.csv',index=False)
    fig.savefig(fig_dir+'RQ4_prediction_prob.png')



####### Helper Functions #####
###############################################################################
def get_rule_str_of_rulefit(local_model, X_explain, debug= False):
    rules = local_model.get_rules() # fetch the list of all the rules for this particular instance 
    rules = rules[(rules['type']=='rule') & (rules['coef'] > 0) & (rules['importance'] > 0)]
    rules_list = list(rules['rule'])
    
    rule_eval_result = []
    i = 0
    for r in rules_list:
        if debug and i < 3 : 
            print(r, type(r))
            i += 1
        py_exp_pred = eval_rule(r, X_explain)[0]
        rule_eval_result.append(py_exp_pred)
          
    rules['is_satisfy_instance'] = rule_eval_result

    rules = rules[rules['is_satisfy_instance']==True]

    rules = rules.sort_values(by='importance', ascending=False)
    
    rule_str = rules.iloc[0]['rule']
    
    return rule_str

def get_rule_str_of_nodeharvest(local_model, X_explain, debug = False):
    rules = local_model.get_rules() # fetch the list of all the rules for this particular instance 

    rules_list = list(rules['rule'])
    rule_eval_result = []
    i = 0
    for r in rules_list:
        if debug : 
            if i < 3 : 
                print("level 1", r, type(r), len(r))
            i += 1
        if len(r) == 0 : 
            rule_eval_result.append(False)
            continue
        try : 
            if i > 11 and i < 14 : 
                nh_exp_pred = eval_rule(r, X_explain,debug)[0]
            else :
                nh_exp_pred = eval_rule(r, X_explain)[0]
        except :
            print(i , r, type(r), len(r))
        rule_eval_result.append(nh_exp_pred)     
    rules['is_satisfy_instance'] = rule_eval_result
    rules = rules[rules['is_satisfy_instance']==True]

    rules = rules.sort_values(by='importance', ascending=False)
    rule_str = rules.iloc[0]['rule']
    return rule_str

def eval_rule(rule, x_df, debug=False):
    var_in_rule = list(set(re.findall('[a-zA-Z]+', rule)))
    
    rule = re.sub(r'\b=\b','==',rule)
    if 'or' in var_in_rule:
        var_in_rule.remove('or')
        
    rule = rule.replace('&','and')
    
    if (debug) : 
        print(rule)
    
    eval_result_list = []

    for i in range(0,len(x_df)):
        x = x_df.iloc[[i]]
        col = x.columns
        var_dict = {}

        for var in var_in_rule:
            var_dict[var] = float(x[var])
        
        eval_result = eval(rule,var_dict)
        eval_result_list.append(eval_result)

        
    return eval_result_list

def summarize_rule_eval_result(rule_str, x_df):
    all_eval_result = eval_rule(rule_str, x_df)
    all_eval_result = np.array(all_eval_result).astype(bool)

    return all_eval_result


####### Authors RQ3 Evaluation #####
###############################################################################
def rq3_eval(proj_name, global_model_name, debug = False):
    global_model, correctly_predict_df, indep, dep, feature_df = prepare_data_for_testing(proj_name, global_model_name)
    x_test, y_test = prepare_data(proj_name, mode = 'test')

    rq3_explanation_result = pd.DataFrame()
    
    pyexp_guidance_result_list = []
    lime_guidance_result_df = pd.DataFrame()
    
    for i in tqdm(range(0,len(feature_df))):
        if debug : 
            if i > 1 : 
                break 
        
        X_explain = feature_df.iloc[[i]]

        row_index = str(X_explain.index[0])

        exp_obj = pickle.load(open(os.path.join(exp_dir,proj_name,global_model_name,'all_explainer_'+row_index+'.pkl'),'rb'))
        py_exp = exp_obj['pyExplainer']
        lime_exp = exp_obj['LIME']
        nh_exp = exp_obj['MBase']

        # load local models
        py_exp_local_model = py_exp['local_model']
        lime_exp_local_model = lime_exp['local_model']
        nh_exp_local_model = nh_exp['local_model']
        
        # generate explanations                
        py_exp_the_best_defective_rule_str = get_rule_str_of_rulefit(py_exp_local_model, X_explain, debug)
        lime_the_best_defective_rule_str = lime_exp['rule'].as_list()[0][0]
        nh_exp_the_best_defective_rule_str = get_rule_str_of_nodeharvest(nh_exp_local_model, X_explain, debug)

        if debug :
            print(i)
            print(py_exp_the_best_defective_rule_str) 
            print(lime_the_best_defective_rule_str)
            print(nh_exp_the_best_defective_rule_str)
            break 

        # check whether explanations apply to the instance to be explained
        py_exp_pred = eval_rule(py_exp_the_best_defective_rule_str, X_explain)[0]
        lime_pred = eval_rule(lime_the_best_defective_rule_str, X_explain)[0]
        nh_exp_pred = eval_rule(nh_exp_the_best_defective_rule_str, X_explain)[0] 

        ################### Specific to PyExplainer ##############################
        condition_list = py_exp_the_best_defective_rule_str.split('&')

        for condition in condition_list:
            condition = condition.strip()

            py_exp_rule_eval = summarize_rule_eval_result(condition, x_test)

            rule_rec = recall_score(y_test, py_exp_rule_eval)

            py_exp_serie_test = pd.Series(data=[proj_name, row_index, 'pyExplainer',global_model_name, condition, rule_rec])
            rq3_explanation_result = rq3_explanation_result.append(py_exp_serie_test,ignore_index=True)

        ################### Specific to PyExplainer Ends ##############################
        
        ################### Specific to LIME Starts ##############################
        lime_rule_eval = summarize_rule_eval_result(lime_the_best_defective_rule_str, x_test)

        rule_rec = recall_score(y_test, lime_rule_eval)

        lime_serie_test = pd.Series(data=[proj_name, row_index, 'LIME',global_model_name, lime_the_best_defective_rule_str, rule_rec])
        rq3_explanation_result = rq3_explanation_result.append(lime_serie_test,ignore_index=True)
        ################### Specific to LIME Ends ##############################
        

        ################### Specific to Nodeharvest Starts ##############################
        condition_list = nh_exp_the_best_defective_rule_str.split('&')

        for condition in condition_list:
            condition = condition.strip()

            nh_exp_rule_eval = summarize_rule_eval_result(condition, x_test)

            rule_rec = recall_score(y_test, nh_exp_rule_eval)

            nh_exp_serie_test = pd.Series(data=[proj_name, row_index, 'mBase',global_model_name, condition, rule_rec])
            rq3_explanation_result = rq3_explanation_result.append(nh_exp_serie_test,ignore_index=True)
        ################### Specific to Nodeharvest Ends ##############################

    if debug : 
        return 1
    
    rq3_explanation_result.columns = ['project','commit_id','method','global_model','explanation','recall']
    rq3_explanation_result.to_csv(result_dir+'RQ3_'+proj_name+'_'+global_model_name+'_explanation_eval_split_rulefit_condition.csv',
                                  index=False)
    
def get_percent_unique_explanation(proj_name, global_model_name, agnostic_name , explanation_list):
    
    print('project: {}, JIT model: {}, Agnostic: {}'.format(proj_name, global_model_name,agnostic_name))
    total_exp = len(explanation_list)
    total_unique_exp = len(set(explanation_list))
    percent_unique = (total_unique_exp/total_exp)*100
    
    # Not the best way, but yes ok counts for the maximum occurence only. 
    count_exp = Counter(explanation_list)
    max_exp_count = max(list(count_exp.values()))
    percent_dup_explanation = (max_exp_count/total_exp)*100
    
    print('% unique explanation is',round(percent_unique,2))
    print('% duplicate explanation is', round(percent_dup_explanation))
    
    print('-'*50)

def printRQ3Scores(results,proj_name, global_model_name) : 

    r_lime = results[results['method']=='LIME']
    r_pyexp = results[results['method']=='pyExplainer']
    r_nhexp = results[results['method']=='mBase']

    get_percent_unique_explanation(proj_name,global_model_name, 'LIME', list(r_lime['explanation']))
    get_percent_unique_explanation(proj_name,global_model_name, 'pyExplainer', list(r_pyexp['explanation']))
    get_percent_unique_explanation(proj_name,global_model_name, 'NodeHarvest', list(r_nhexp['explanation']))

def show_rq3_eval_result():

    openstack_rf = pd.read_csv(result_dir+'RQ3_openstack_RF_explanation_eval_split_rulefit_condition.csv')
    qt_rf = pd.read_csv(result_dir+'RQ3_qt_RF_explanation_eval_split_rulefit_condition.csv')
    result_rf = pd.concat([openstack_rf, qt_rf])
    result_rf['global_model'] = 'RF'
    
    openstack_lr = pd.read_csv(result_dir+'RQ3_openstack_LR_explanation_eval_split_rulefit_condition.csv')
    qt_lr = pd.read_csv(result_dir+'RQ3_qt_LR_explanation_eval_split_rulefit_condition.csv')
    result_lr = pd.concat([openstack_lr, qt_lr])
    result_lr['global_model'] = 'LR'
    
    all_result = pd.concat([result_rf, result_lr])
    all_result['recall'] = all_result['recall']*100
    openstack_result = all_result[all_result['project']=='openstack']
    qt_result = all_result[all_result['project']=='qt']
    
    fig, axs = plt.subplots(1,2, figsize=(10,10))

    axs[0].set_title('Openstack')
    axs[1].set_title('Qt')

    my_pal = {"pyExplainer": "y", "LIME": "b", "mBase":"g"}
    sns.boxplot(data=openstack_result, x='global_model', y='recall', 
                hue='method', ax=axs[0]).set(xlabel='', ylabel='Consistency Percentage (%)')
    sns.boxplot(data=qt_result, x='global_model', y='recall', 
                hue='method', ax=axs[1]).set(xlabel='', ylabel='')

    plt.show()
    fig.savefig(fig_dir+'RQ3.png')
    ## by now the figure has been drawn, 
    # now I am reporting the numbers for Agnostics

    printRQ3Scores(openstack_rf ,'OpenStack', 'RF') 
    printRQ3Scores(openstack_lr ,'OpenStack', 'LR') 
    printRQ3Scores(qt_rf ,'qt', 'RF') 
    printRQ3Scores(qt_lr ,'qt', 'LR') 
    

####### Helper Functions #####
###############################################################################
def flip_rule(rule,debug=False):
    rule = re.sub(r'\b=\b',' = ',rule) # for LIME
    found_rule = re.findall('.* <=? [a-zA-Z]+ <=? .*', rule) # for LIME
    ret = ''
    
    # for LIME that has condition like this: 0.53 < nref <= 0.83
    if len(found_rule) > 0:
        found_rule = found_rule[0]
    
        var_in_rule = re.findall('[a-zA-Z]+',found_rule)

        var_in_rule = var_in_rule[0]
        
        splitted_rule = found_rule.split(var_in_rule)
        splitted_rule[0] = splitted_rule[0] + var_in_rule # for left side
        splitted_rule[1] = var_in_rule + splitted_rule[1] # for right side
        combined_rule = splitted_rule[0] + ' or ' + splitted_rule[1]
        ret = flip_rule(combined_rule)
        
    else:
        for tok in rule.split():
            if tok in flip_sign_dict:
                ret = ret + flip_sign_dict[tok] + ' '
            else:
                ret = ret + tok + ' '
    return ret

def get_combined_probs(X_input, global_model, input_guidance, input_SD,debug=False):

    k_percent = 10
    prob_og = global_model.predict_proba(X_input)[0][1]


    output_df = pd.DataFrame()
    # 1) Revised values must not be negative -> if val < 0 then val = 0
    # 2) Revised values must not be lesser/greater than the actual values for < / > operations
    # E.g., {LOC < 50} => Clean and the actual LOC is 30: 
    # For a 10-percent decrease approach, the revised value according to the threshold is 45
    # In this case (Actual < Revised), we apply the 10-percent decrease approach to the actual value instead of the threshold.
    # Thus, the final revised value is 27 (10-percent decrease from 30)
    # There are 2 approaches:
    # (1) n-percent increase/decrease, e.g., 10-percent
    # (2) an SD_train increase/decrease, e.g., 1SD

    X_revised_sd = X_input.copy()
    for guidance_i in input_guidance.split('&'):
        tmp_g = guidance_i.strip().split(' ')
        if len(tmp_g) != 3:
            continue

        revised_var = tmp_g[0]    
        revised_opr = tmp_g[1] 
        actual_val = X_input[revised_var][0]

        if '<' in revised_opr:
            # < or <=
            revised_from_threshold = float(tmp_g[2]) - input_SD[revised_var]
            revised_from_actual = X_input[revised_var][0] - input_SD[revised_var]
            # the revised value from threshold must not greater than the actual values
            if revised_from_threshold > actual_val:
                actual_revised_value = revised_from_actual
            else:
                actual_revised_value = revised_from_threshold
        else:
            # > or >=
            revised_from_threshold = float(tmp_g[2]) + input_SD[revised_var]
            revised_from_actual = X_input[revised_var][0] + input_SD[revised_var]
            # the revised value from threshold must not less than the actual values
            if revised_from_threshold < actual_val:
                actual_revised_value = revised_from_actual
            else:
                actual_revised_value = revised_from_threshold

        # the revised value must not be negative
        if actual_revised_value < 0:
            actual_revised_value = 0

        X_revised_sd[revised_var] = actual_revised_value

#     prob_revised_percent = global_model.predict_proba(X_revised_percent)[0][1]
    prob_revised_sd = global_model.predict_proba(X_revised_sd)[0][1]
    
    tmp_out = pd.Series(data=[input_guidance, prob_og, prob_revised_sd])
    output_df = output_df.append(tmp_out,ignore_index=True)
    if len(output_df) == 0:
        return []
    output_df.columns = ['guidance', 'probOg', 'probRevisedSD']
    return output_df



####### Authors What-If Evaluation #####
###############################################################################
def what_if_analysis(proj_name, global_model_name, debug = False):
    global_model, correctly_predict_df, indep, dep, feature_df = prepare_data_for_testing(proj_name, global_model_name)

    x_train, x_test, y_train, y_test = prepare_data(proj_name, mode = 'all')

    rq3_explanation_result = pd.DataFrame()

    pyexp_guidance_result_list = []

    x_train_sd = x_train.std()
    
    all_df = pd.DataFrame()

    for i in tqdm(range(0,len(feature_df))):
        if debug and i>0 : 
            break 
        tmp_df = pd.DataFrame()
        tmp_df2 = pd.DataFrame()

        X_explain = feature_df.iloc[[i]]

        row_index = str(X_explain.index[0])

        exp_obj = pickle.load(open(os.path.join(exp_dir,proj_name,global_model_name,'all_explainer_'+row_index+'.pkl'),'rb'))
        py_exp = exp_obj['pyExplainer']
        lime_exp = exp_obj['LIME']
        nh_exp = exp_obj['MBase']

        # load local models
        py_exp_local_model = py_exp['local_model']
        lime_exp_local_model = lime_exp['local_model']
        nh_exp_local_model = nh_exp['local_model']

        # generate explanations                
        py_exp_the_best_defective_rule_str = get_rule_str_of_rulefit(py_exp_local_model, X_explain)
        lime_the_best_defective_rule_str = lime_exp['rule'].as_list()[0][0]
        nh_exp_the_best_defective_rule_str = get_rule_str_of_nodeharvest(nh_exp_local_model, X_explain)

        
        # generate guidance
        pyFlip = flip_rule(py_exp_the_best_defective_rule_str)

        # if debug : 
        #     print("about to generate flip rule for nh")
        nhFlip = flip_rule(nh_exp_the_best_defective_rule_str,debug)
        # if debug : 
        #     print("flip rule successfully generated ")


        tmp_df_i = get_combined_probs(X_explain, global_model, pyFlip, x_train_sd)
        if len(tmp_df_i) > 0:
            tmp_df_i['gtype'] = 'pyFlip'
            tmp_df = tmp_df.append(tmp_df_i)
            tmp_df['commit_id'] = row_index
            tmp_df['model'] = global_model_name
            tmp_df['project'] = proj_name
            all_df = all_df.append(tmp_df)

        if debug : 
            print("about to generate combined probs for nh")

        tmp_df_j = get_combined_probs(X_explain, global_model, nhFlip, x_train_sd, debug)

        if debug : 
            print("combined probs successfully generated ") 

        if len(tmp_df_j) > 0:
            tmp_df_j['gtype'] = 'nhFlip'
            tmp_df2 = tmp_df2.append(tmp_df_j)
            tmp_df2['commit_id'] = row_index
            tmp_df2['model'] = global_model_name
            tmp_df2['project'] = proj_name
            all_df = all_df.append(tmp_df2)
        
        if debug : 
            print("df successfully generated ")
        # print('finished {}/{} commits'.format(str(i+1), str(len(feature_df))))

    print('finished what-if of',proj_name)
    all_df.to_csv(result_dir+proj_name+'_'+global_model_name+'_combined_prob_from_guidance.csv',index=False)

def show_what_if_eval_result():
    openstack_rf = pd.read_csv(result_dir+'openstack_RF_combined_prob_from_guidance.csv')
    qt_rf = pd.read_csv(result_dir+'qt_RF_combined_prob_from_guidance.csv')
    result_rf = pd.concat([openstack_rf, qt_rf])

    openstack_lr = pd.read_csv(result_dir+'/openstack_LR_combined_prob_from_guidance.csv')
    qt_lr = pd.read_csv(result_dir+'/qt_LR_combined_prob_from_guidance.csv')
    result_lr = pd.concat([openstack_lr, qt_lr])

    all_result = pd.concat([result_rf, result_lr])

    all_result['predOg'] = all_result['probOg'] >= 0.5
    all_result['predSD'] = all_result['probRevisedSD'] >= 0.5

    pred_og = list(all_result['predOg'])
    pred_sd = list(all_result['predSD'])

    compare_list = []

    for a,b in zip(pred_og, pred_sd):
        compare_list.append(a!=b)

    all_result['isFlip'] = compare_list
    all_result['prob_diff'] = (all_result['probOg']-all_result['probRevisedSD'])*100

    all_result_prob_change = all_result[all_result['prob_diff']>=0]
    

    openstack_result_prob_change = all_result_prob_change[all_result_prob_change['project']=='openstack']
    qt_result_prob_change = all_result_prob_change[all_result_prob_change['project']=='qt']

    df_list = list(all_result_prob_change.groupby(['project','model']))

    openstack_lr = df_list[0][1]
    openstack_rf = df_list[1][1]
    qt_lr = df_list[2][1]
    qt_rf = df_list[3][1]

    openstack_lr_py = openstack_lr[openstack_lr['gtype'] == 'pyFlip']
    openstack_rf_py = openstack_rf[openstack_rf['gtype'] == 'pyFlip']
    openstack_lr_nh = openstack_lr[openstack_lr['gtype'] == 'nhFlip']
    openstack_rf_nh = openstack_rf[openstack_rf['gtype'] == 'nhFlip']

    qt_rf_py = qt_rf[qt_rf['gtype'] == 'pyFlip']
    qt_lr_py = qt_lr[qt_lr['gtype'] == 'pyFlip']
    qt_rf_nh = qt_rf[qt_rf['gtype'] == 'nhFlip']
    qt_lr_nh = qt_lr[qt_lr['gtype'] == 'nhFlip']

    openstack_rf_reverse_percent_py = np.mean(list(openstack_rf_py['isFlip']))*100
    openstack_lr_reverse_percent_py = np.mean(list(openstack_lr_py['isFlip']))*100
    qt_rf_reverse_percent_py = np.mean(list(qt_rf_py['isFlip']))*100
    qt_lr_reverse_percent_py = np.mean(list(qt_lr_py['isFlip']))*100

    openstack_rf_reverse_percent_nh = np.mean(list(openstack_rf_nh['isFlip']))*100
    openstack_lr_reverse_percent_nh = np.mean(list(openstack_lr_nh['isFlip']))*100
    qt_rf_reverse_percent_nh = np.mean(list(qt_rf_nh['isFlip']))*100
    qt_lr_reverse_percent_nh = np.mean(list(qt_lr_nh['isFlip']))*100

    openstack_reverse_percent_df = pd.DataFrame()
    openstack_reverse_percent_df = openstack_reverse_percent_df.append(pd.Series(['RF', 'pyExplainer' , openstack_rf_reverse_percent_py]), ignore_index=True)
    openstack_reverse_percent_df = openstack_reverse_percent_df.append(pd.Series(['LR', 'pyExplainer' , openstack_lr_reverse_percent_py]), ignore_index=True)
    openstack_reverse_percent_df = openstack_reverse_percent_df.append(pd.Series(['RF', 'mBase', openstack_rf_reverse_percent_nh]), ignore_index=True)
    openstack_reverse_percent_df = openstack_reverse_percent_df.append(pd.Series(['LR', 'mBase', openstack_lr_reverse_percent_nh]), ignore_index=True)
    openstack_reverse_percent_df.columns = ['model','method', '% reverse']

    qt_reverse_percent_df = pd.DataFrame()
    qt_reverse_percent_df = qt_reverse_percent_df.append(pd.Series(['RF', 'pyExplainer' ,  qt_rf_reverse_percent_py]), ignore_index=True)
    qt_reverse_percent_df = qt_reverse_percent_df.append(pd.Series(['LR', 'pyExplainer' , qt_lr_reverse_percent_py]), ignore_index=True)
    qt_reverse_percent_df = qt_reverse_percent_df.append(pd.Series(['RF', 'mBase',  qt_rf_reverse_percent_nh]), ignore_index=True)
    qt_reverse_percent_df = qt_reverse_percent_df.append(pd.Series(['LR', 'mBase',  qt_lr_reverse_percent_nh]), ignore_index=True)
    qt_reverse_percent_df.columns = ['model','method','% reverse']

    # plot probability difference
    plt.figure()

    fig, axs = plt.subplots(1,2)

    fig.suptitle('Probability difference between actual prediction and simulated prediction')
    axs[0].set(ylim=(0, 100))
    axs[1].set(ylim=(0, 100))

    axs[0].set_title('openstack')
    axs[1].set_title('qt')

    sns.boxplot(x='model',y='prob_diff', data=openstack_result_prob_change, hue='gtype', ax=axs[0])
    sns.boxplot(x='model',y='prob_diff', data=qt_result_prob_change, hue='gtype', ax=axs[1])

    plt.show()

    fig.savefig(fig_dir+'what_if_prob_diff.png')

    # plot percentage of reversed predictions
    plt.figure()

    fig, axs = plt.subplots(1,2)

    fig.suptitle('Percentage of defective commits that their predictions are inversed')

    axs[0].set(ylim=(0, 100))
    axs[1].set(ylim=(0, 100))

    axs[0].set_title('openstack')
    axs[1].set_title('qt')

    sns.barplot(x='model',y='% reverse', data=openstack_reverse_percent_df, hue='method', ax=axs[0])
    sns.barplot(x='model',y='% reverse', data=qt_reverse_percent_df, hue='method', ax=axs[1])

    for index, row in openstack_reverse_percent_df.iterrows():
        axs[0].text(row.name,row['% reverse'], round(row['% reverse'],2), color='black', ha="center")

    for index, row in qt_reverse_percent_df.iterrows():
        axs[1].text(row.name,row['% reverse'], round(row['% reverse'],2), color='black', ha="center")

    plt.show()

    fig.savefig(fig_dir+'what_if_percent_reverse_prediction.png')
    
