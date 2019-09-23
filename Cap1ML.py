import csv 
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pylab import savefig
from datetime import datetime
import pandas_profiling
import sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def write_log_header(working_df, data_filename, session_log) :
    """save summary information to results file
    """

    cols_with_nulls = df.columns[df.isnull().any()]
    with open(session_log, 'w+') as f:   
        f.write('\nsklearn version: ' + sklearn.__version__)
        f.write('\npandas version: ' + pd.__version__)
        f.write('\nnumpy version: ' + np.__version__)
        f.write('\nPandas Profile version: ' + pandas_profiling.__version__)

        f.write('\n\nData file ' + data_filename + '\n\n')
        f.write('Shape: ' + str(working_df.shape) + '\n')
        f.write('Columns with null data: \n')
        if len(cols_with_nulls) == 0:
            f.write('None\n')
        else:
            for col in cols_with_nulls:
                f.write(col + '\n')

        cols = sorted(df.columns)
        f.write('\nColumns in data file: \n')
        for item in cols:
            f.write(item + '\n')            


def update_columns(working_df, col_info_filename):
    """ update column names to readable versions, update column types
    """
    info_dict = pd.read_csv(col_info_filename, index_col=0, squeeze=True, header=None).to_dict()
    
    #these are numerical columns and are interpreted as such
    #working_df['SMOKAGEREG'] = pd.to_numeric(working_df['SMOKAGEREG'])
    #working_df['MOD10FWK'] = pd.to_numeric(working_df['MOD10FWK'])
    #working_df['VIG10FWK'] = pd.to_numeric(working_df['VIG10FWK'])
    #working_df['STRONGFWK'] = pd.to_numeric(working_df['STRONGFWK'])
    
    #working_df['CPOXEV'] = working_df['CPOXEV'].astype('category')

    return working_df.rename(columns=info_dict)


def drop_columns(working_df, cols_to_drop_filename, session_log):
    """drop columns with high percentage of unusable data in target-filtered dataset
    """
    
    with open(session_log, 'a') as f:   
        f.write('\nShape before dropping unusable columns\n' + str(working_df.shape))
   
    #read columns from file
    with open(cols_to_drop_filename, 'r') as f_cols:
        reader = csv.reader(f_cols)
        cols_to_drop = [r[0] for r in reader]
    
    #drop columns
    working_df.drop(columns=cols_to_drop, axis=1, inplace=True)
    
    with open(session_log, 'a') as f:   
        f.write('\nShape after dropping unusable columns\n' + str(working_df.shape))
    
    return working_df


def drop_invalid_target_rows(working_df, session_log):
    """update dataset to only use rows with target = actual no (=1) or actual yes (=2)
    """
    
    with open(session_log, 'a') as f:   
        f.write('\nShape before deleting invalid target rows\n' + str(working_df.shape))
   
    new_df = working_df[(working_df['HYPERTENEV'] !='0') & \
                        (working_df['HYPERTENEV'] !='8') ]
   
    with open(session_log, 'a') as f:   
        f.write('\nShape after deleting invalid target rows\n' + str(new_df.shape))
        f.write('\nhypertension value counts:\n ')
        counts = new_df['HYPERTENEV'].value_counts()
        print(counts)
        f.write('\nindex: ' + str(counts.index.tolist()))
        f.write('\nvalues: ' + str(counts.values.tolist()))
            
    return new_df

def drop_missing_data_rows(working_df, session_log):
    with open(session_log, 'a') as f:   
        f.write('\nShape before deleting missing data rows\n' + str(working_df.shape))
   
    new_df = working_df[(working_df['CPOXEV'] !='0') & \
                        (working_df['TYPPLSICK'] !='0')]
   
    with open(session_log, 'a') as f:   
        f.write('\nShape after deleting missing data rows\n' + str(new_df.shape))
        f.write('\nhypertension value counts:\n ')
        counts = new_df['HYPERTENEV'].value_counts()
        print(counts)
        f.write('\nindex: ' + str(counts.index.tolist()))
        f.write('\nvalues: ' + str(counts.values.tolist()))
            
    return new_df

def run_Pandas_Profiling(working_df, session_log, Break_into_sections = False, num_cols=5):
    """Run Pandas_Profiling 
    """
    t0 = datetime.now()
    start_col_idx = 0
    end_col_idx = num_cols
    cols = list(working_df.columns)
    total_cols = len(cols)
    
    with open(session_log, 'a') as f:   
        f.write('\n\nrun_Pandas_Profiling Begin\n')
    
    if Break_into_sections:
        while end_col_idx <= total_cols:
            #slice the column list by the given num_cols
            cols_by_section = cols[start_col_idx:end_col_idx] 

            #make sure the target variable is in the heat maps
            if 'Ever told had hypertension' not in cols_by_section:
                cols_by_section.append('Ever told had hypertension')

            #create a dataset with only the selected columns
            df_to_profile = working_df.loc[:, cols_by_section]

            with open(session_log, 'a') as f:   
               f.write('\nProfiling : ' + str(df_to_profile.columns) + '\n')
                
            #run the profile
            profile = df_to_profile.profile_report(title='Pandas Profiling Report')
            profile.to_file(output_file= working_file_path + timestamp + '_' + str(start_col_idx) + "output.html")

            #update the indicies for the next slice or the end slice
            start_col_idx = end_col_idx 
            if (end_col_idx + num_cols > total_cols) and (total_cols % num_cols > 0):
                end_col_idx += total_cols % num_cols
            else:
                end_col_idx += num_cols
    else:
        #run the profile on all of the columns
        profile = working_df.profile_report(title='Pandas Profiling Report')
        profile.to_file(output_file= working_file_path + timestamp + '_' + str(start_col_idx) + "output.html")


    total_time = datetime.now() - t0
    with open(session_log, 'a') as f:   
        f.write('\nrun_Pandas_Profiling End.  Duration: ' + str(total_time) + '\n')

def calc_Correlations(working_df, session_log):
    """calculate and plot a correlation matrix"""
    t0 = datetime.now()
    with open(session_log, 'a') as f:   
        f.write('\n\ncalc_Correlations Begin\n')
    
    ##categorical columns
    #cols_to_corr = ['Ever told had hypertension', 'Ever told had angina pectoris', 'Ever told had asthma', 'Ever told had cancer']

    #categorical and incorrectly classified as numeric columns
    cols_to_corr = ['Ever told had hypertension', 'Ever had chickenpox', 'Hispanic ethnicity', 'Legal marital status']
    
    sub_df = working_df.loc[:, cols_to_corr]
    df_cm = sub_df.corr(method ='spearman')
    svm = sn.heatmap(df_cm, annot=True,cmap='coolwarm', linecolor='white', linewidths=1)
    plt.savefig('svm_conf.png', dpi=400)

    total_time = datetime.now() - t0
    with open(session_log, 'a') as f:   
        f.write('\ncalc_Correlations End. Duration: ' + str(total_time) + '\n')

def KNN_model(X_train_data, X_test_data, y_train_data, y_test_data):
    """Use a k-NN model to predict target
    """
    neighbors = 6
    print('KNN Model, neighbors= ', neighbors)
    knn = KNeighborsClassifier(neighbors)

    # Fit the classifier to the data
    knn.fit(X_train, y_train)
    
    # Predict the labels for the training data X
    y_pred = knn.predict(X_test)
    
    # Print the accuracy
    print('KNN Score: ', knn.score(X_test, y_test))

    # Print the confusion matrix and classification report
    print('Confusion maxtrix: \n', confusion_matrix(y_test, y_pred))
    print('Classification report: \n', classification_report(y_test, y_pred))
    

def log_reg_model(X_train_data, X_test_data, y_train_data, y_test_data):
    """Use logistic regression model to predict target
    """   
    print('\nLogistic Regression Model\n')
    logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('Confusion maxtrix: \n', confusion_matrix(y_test, y_pred))
    print('Classification report: \n', classification_report(y_test, y_pred))
    
    cols_lst = X_train_data.columns
    coefs = list(zip(cols_lst, logreg.coef_[0]))
    coefs.sort(key=lambda tup: tup[1])
    print('Coefficients: \n')
    for col, value in coefs:
        print('{} : {}'.format(value, col))

def random_forest_classifier(X_train_data, X_test_data, y_train_data, y_test_data):
    """Use random forst classifer to predict target
    """
    print('\nRandom Forest Model\n')
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
 
    print('Confusion maxtrix: \n', confusion_matrix(y_test, y_pred))
    print('Classification report: \n', classification_report(y_test, y_pred))

def AdaBoost_classifer(X_train_data, X_test_data, y_train_data, y_test_data):
    """Use random forst classifer to predict target
    """
    print('\nAdaBoost Model\n')
    #svc=SVC(probability=True, kernel='linear')
    #abc = AdaBoostClassifier(n_estimators=10, base_estimator=svc, learning_rate=1)
    abc = AdaBoostClassifier(n_estimators=10, base_estimator=None, learning_rate=1)
    model = abc.fit(X_train_data, y_train_data)
    y_pred = model.predict(X_test_data)
    print('Confusion maxtrix: \n', confusion_matrix(y_test, y_pred))
    print('Classification report: \n', classification_report(y_test, y_pred))




if __name__ == '__main__': 
    t0 = datetime.now()
    print("working!")
   
    working_file_path = '/Users/alexia/Documents/Springboard/Capstone1/Cap1testing/'
    working_file = working_file_path + 'nhis_00010.csv'
    flag_col_file = working_file_path + 'nhis_flag_cols.csv'
    col_info_file = working_file_path + 'Extract9ColMapping.csv'
    cols_to_drop_file = working_file_path + 'Extract9ColsToDrop.csv'
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = working_file_path + 'results' + timestamp + '.txt'
    final_file = working_file_path + 'finaldf_' + timestamp + '.csv'

    df = pd.read_csv(working_file, dtype='category', index_col = 'YEAR')#, nrows=1000)
    #df = pd.read_csv(working_file, index_col = 'YEAR') #, nrows=1000)

    write_log_header(df, working_file, log_file)

    df = drop_invalid_target_rows(df, log_file)

    df = drop_missing_data_rows(df, log_file)

    df = drop_columns(df, cols_to_drop_file, log_file)

    df = update_columns(df, col_info_file)
    
    df.to_csv(final_file)

    #calc_Correlations(df, log_file)

    #have profile reports for full dataset, in batches of 10
    #run_Pandas_Profiling(df, log_file, True, 10)


    print("Training and testing data")
    ## Create arrays for the features and the response variable
    #y = df['Ever told had hypertension'].cat.codes
    y = df['Ever told had hypertension']
    X = df.drop('Ever told had hypertension', axis=1)
    
    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    print('#train rows, cols ', X_train.shape)

    #KNN_model(X_train, X_test, y_train, y_test)

    #log_reg_model(X_train, X_test, y_train, y_test)
    
    #random_forest_classifier(X_train, X_test, y_train, y_test)

    #AdaBoost_classifer(X_train, X_test, y_train, y_test)
    
    print("Done!")
    total = datetime.now() - t0
    print("time taken: ", total)
    with open(log_file, 'a') as f:   
        f.write('\nReview time = ' + str(total))

