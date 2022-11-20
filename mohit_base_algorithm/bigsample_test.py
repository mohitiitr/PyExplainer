# Authors: Martin Billinger <flkazemakase@gmail.com>
#
# Licence: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from nodeharvest import NodeHarvest
from rulefit import RuleFit

print("start")
n = 100
plot_step = 0.02
solver = 'scipy_robust'     # 'cvx_robust' is faster, but requires cvxopt installed
np.random.seed(42)


def model(x, e=0.25):
    return np.prod(np.sin(2 * np.pi * x), axis=1) + np.random.randn(x.shape[0]) * e


x = np.random.rand(n, 2)
y = model(x)

rf = RandomForestRegressor(n_estimators=50, max_depth=3, min_samples_leaf=3)
rf.fit(x, y)

nh = NodeHarvest(max_nodecount=None, solver=solver, verbose=True)
nh.fit(rf, x, y)

rff = RuleFit(rfmode='regress',exp_rand_tree_size=False, random_state=0, max_rules=200, cv=5, max_iter=30, n_jobs=-1)
rff.fit(x,y)
rules = rff.get_rules()
rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
rules = rules[rules.type == 'rule']
nrules = rules.shape[0]

n_nodes = nh.coverage_matrix_.shape[1]
n_selected = np.sum(nh.get_weights() > 0)


xx, yy = np.meshgrid(np.arange(0, 1 + plot_step, plot_step), np.arange(0, 1 + plot_step, plot_step))
x_test = np.c_[xx.ravel(), yy.ravel()]

plt.figure()
z = rf.predict(x_test)
mse = np.var(z - model(x_test))
plt.contourf(xx, yy, z.reshape(xx.shape))
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.title('Random Forest: %d nodes, MSE=%f' % (n_nodes, mse))

plt.figure()
z = nh.predict(x_test)
mse = np.var(z - model(x_test))
plt.contourf(xx, yy, z.reshape(xx.shape))
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.title('Node Harvest: %d nodes, MSE=%f' % (n_selected, mse))

plt.figure()
z = rff.predict(x_test)
mse = np.var(z - model(x_test))
plt.contourf(xx, yy, z.reshape(xx.shape))
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.title('Rulefit : %d Rules, MSE=%f' % (nrules, mse))

plt.show()


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

from pyexplainer.pyexplainer_pyexplainer import *

sys.path.append(os.path.abspath('../'))
from mohit_base_algorithm.pyexplainer_pyexplainer import *

import warnings
warnings.filterwarnings("ignore")

data_path = './dataset/'
result_dir = './eval_result/'
dump_dataframe_dir = './prediction_result/'
exp_dir = './explainer_object/'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    
if not os.path.exists(dump_dataframe_dir):
    os.makedirs(dump_dataframe_dir)
    
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

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

def create_explainer(proj_name, global_model_name, x_train, x_test, y_train, y_test, df_indices):
    
    save_dir = os.path.join(exp_dir,proj_name,global_model_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    global_model = pickle.load(open(proj_name+'_'+global_model_name+'_global_model.pkl','rb'))

    indep = x_test.columns
    dep = 'defect'
    class_label = ['clean', 'defect']
    

    ##################################
    # # for our apporach
    # pyExp = PyExplainer(x_train, y_train, indep, dep, global_model, class_label)

    # # for baseline
    # # note: 6 is index of 'self' feature
    # lime_explainer = LimeTabularExplainer(x_train.values, categorical_features=[6],
    #                                   feature_names=indep, class_names=class_label, 
    #                                   random_state=0)
    ##################################

    # for my testing 
    mBase = MohitBase(x_train, y_train, indep, dep, global_model, class_label)

    feature_df = x_test.loc[df_indices]
    test_label = y_test.loc[df_indices]
    
    for i in range(0,len(feature_df)):
        X_explain = feature_df.iloc[[i]]
        y_explain = test_label.iloc[[i]]

        row_index = str(X_explain.index[0])

        # Mohit Base testing
        print("\t starting Mohit-base")
        mBase_obj = mBase.explain(X_explain,
                                   y_explain,
                                   search_function = 'CrossoverInterpolation')
        print("\t done Mohit-base")

        mBase_obj['commit_id'] = row_index

        # because this error is done by authors of actual paper. 
        mBase_obj['local_model'] = mBase_obj['local_rulefit_model']
        del mBase_obj['local_rulefit_model']

        # # Commenting the Part with PyExplainer, 
        # #################################
        # print("\t starting pyexplainer")
        # pyExp_obj = pyExp.explain(X_explain,
        #                            y_explain,
        #                            search_function = 'CrossoverInterpolation')
        # print("\t done pyexplainer")

        # pyExp_obj['commit_id'] = row_index

        # # because I don't want to change key name in another evaluation file
        # pyExp_obj['local_model'] = pyExp_obj['local_rulefit_model']
        # del pyExp_obj['local_rulefit_model']
        # #################################
        
        # # Commenting the part with lime.
        # ################################# 
        # print("\t starting lime")
        # X_explain = feature_df.iloc[i] # to prevent error in LIME
        # exp, synt_inst, synt_inst_for_local_model, selected_feature_indices, local_model = lime_explainer.explain_instance(X_explain, global_model.predict_proba, num_samples=5000)

        # lime_obj = {}
        # lime_obj['rule'] = exp
        # lime_obj['synthetic_instance_for_global_model'] = synt_inst
        # lime_obj['synthetic_instance_for_lobal_model'] = synt_inst_for_local_model
        # lime_obj['local_model'] = local_model
        # lime_obj['selected_feature_indeces'] = selected_feature_indices
        # lime_obj['commit_id'] = row_index
        # print("\t done lime")
        # #################################

        # load already trained file, from previous executions. 
        all_explainer = pickle.load(open(save_dir+'/all_explainer_'+row_index+'.pkl','rb'))

        # update the object
        all_explainer = {'pyExplainer':all_explainer['pyExplainer'], 'LIME': all_explainer['LIME'] , 'MBase' : mBase_obj}
        
        # write the updated object. 
        pickle.dump(all_explainer, open(save_dir+'/all_explainer_'+row_index+'.pkl','wb'))
        
        print('finished {}/{} commits'.format(str(i+1), str(len(feature_df))))

def train_global_model_runner(proj_name, global_model_name):
    x_train, x_test, y_train, y_test = prepare_data(proj_name, mode = 'all')

    train_global_model(proj_name, x_train, y_train,global_model_name)
    print('train {} of {} finished'.format(global_model_name, proj_name))

    
def train_explainer(proj_name, global_model_name):
    x_train, x_test, y_train, y_test = prepare_data(proj_name, mode = 'all')

    correctly_predict_indice = get_correctly_predicted_defective_commit_indices(proj_name, global_model_name, x_test, y_test)
    correctly_predict_indice = set(correctly_predict_indice)
    create_explainer(proj_name, global_model_name, x_train, x_test, y_train, y_test, correctly_predict_indice)

proj_name = sys.argv[1]
proj_name = proj_name.lower()
global_model = sys.argv[2]
global_model = global_model.upper()

if proj_name not in ['openstack','qt'] or global_model not in ['RF','LR']:
    print('project name must be "openstack" or "qt".')
    print('global model name must be "RF" or "LR".')
    
else:
    print(proj_name, global_model)
    print('training global model')
    train_global_model_runner(proj_name, global_model)
    print('finished training global model')
    
    print('training explainers')
    train_explainer(proj_name, global_model)
    print('finished training explainers')
