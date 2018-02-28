import pandas as pd
import numpy as np 
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 

def analyze_features(features,df):
    '''
    INPUT:
    features: names of features deemed important for data analysis
    df: dataframe with credit card fraud information

    OUTPUT:
    Creates distribution plots of fraud and not fraud instances for each feature provided.  These graphs
    help the viewer understand if the feature is important in distinguishing fraudulent activity.
    '''

    for ind,feature in enumerate(df[features]):
        sns.distplot(df[feature][df['fraud'] == 1], bins=50)
        sns.distplot(df[feature][df['fraud'] == 0], bins=50)
        plt.title('Histogram of Feature: {}'.format(str(feature)))
        plt.show()

def feature_engineering(df):
    '''
    INPUT:
    df: dataframe with credit card fraud information

    OUTPUT:
    Updated df with engineered features designed to improve machine learning classification

    NOTE:
    Based on graphs from analyze_features, we can more closely examine ranges of each feature where
    fraudulent activity is more likely to occur and can create extra features to improve machine
    learning model classification
    '''

    df['V1_'] = df.V1.map(lambda x: 1 if x < -3 else 0)
    df['V2_'] = df.V2.map(lambda x: 1 if x > 2.5 else 0)
    df['V3_'] = df.V3.map(lambda x: 1 if x < -4 else 0)
    df['V4_'] = df.V4.map(lambda x: 1 if x > 2.5 else 0)
    df['V5_'] = df.V5.map(lambda x: 1 if x < -4.5 else 0)
    df['V6_'] = df.V6.map(lambda x: 1 if x < -2.5 else 0)
    df['V7_'] = df.V7.map(lambda x: 1 if x < -3 else 0)
    df['V9_'] = df.V9.map(lambda x: 1 if x < -2 else 0)
    df['V10_'] = df.V10.map(lambda x: 1 if x < -2.5 else 0)
    df['V11_'] = df.V11.map(lambda x: 1 if x > 2 else 0)
    df['V12_'] = df.V12.map(lambda x: 1 if x < -2 else 0)
    df['V14_'] = df.V14.map(lambda x: 1 if x < -2.5 else 0)
    df['V16_'] = df.V16.map(lambda x: 1 if x < -2 else 0)
    df['V17_'] = df.V17.map(lambda x: 1 if x < -2 else 0)
    df['V18_'] = df.V18.map(lambda x: 1 if x < -2 else 0)
    df['V19_'] = df.V19.map(lambda x: 1 if x > 1.5 else 0)
    df['V21_'] = df.V21.map(lambda x: 1 if x > 0.6 else 0)  
    return df

def create_split(df):
    '''
    INPUT:
    df: dataframe with credit card fraud information
    OUTPUT:
    Creates and returns train_test_split variables for undersampled and full data
    '''

    X = df.loc[:, df.columns != 'fraud']
    y = df.loc[:, df.columns == 'fraud']
   
    num_of_frauds = len(df[df['fraud'] == 1])
    fraud_indices = np.array(df[df['fraud'] == 1].index)

    not_fraud_indices = df[df['fraud'] == 0].index

    random_not_fraud_indices = np.random.choice(not_fraud_indices, num_of_frauds, replace = False)
    random_not_fraud_indices = np.array(random_not_fraud_indices)

    under_sample_indices = np.concatenate([fraud_indices,random_not_fraud_indices])

    under_sample_data = df.iloc[under_sample_indices,:]

    X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'fraud']
    y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'fraud']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

    X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample,
                                                                                                        y_undersample,
                                                                                                        test_size = 0.3,
                                                                                                        random_state = 0)

    return X_train, X_test, y_train, y_test, X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample

def printing_Kfold_scores(x_train_data,y_train_data):
    '''
    INPUT:
    train_data: train data used for KFold cross validation.

    OUTPUT:
    best_c: c value that creates best recall score for the classification algorithm
    '''
    fold = KFold(len(y_train_data),5,shuffle=False) 

    c_params = [0.01,0.1,1,10,100]

    results = pd.DataFrame(index = range(len(c_params),2), columns = ['C_parameter','Mean recall score'])
    results['C_parameter'] = c_params

    j = 0
    for c_param in c_params:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')

        recalls = []
        for iteration, indices in enumerate(fold,start=1):

            lr = LogisticRegression(C = c_param, penalty = 'l1')
            lr.fit(x_train_data.iloc[indices[0],:],y_train_data.iloc[indices[0],:].values.ravel())

            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)

            recall = recall_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample)
            recalls.append(recall)
            print('Iteration ', iteration,': recall score = ', recall)

        results.loc[j,'Mean recall score'] = np.mean(recalls)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recalls))
        print('')

    best_c = results[results['Mean recall score'] == results['Mean recall score'].max()]['C_parameter'][0]
    
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')
    
    return best_c

def create_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='Greens'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    else:
        1

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_confusion_matrix(X_train,y_train,X_test,y_test,best_c):
    '''
    INPUT:
    Undersampled train variables for training regression model
    Test variables to analyze effectiveness of model in undersampled or full data analysis

    OUTPUT:
    Confusion matrix graph and recall score associated with model
    '''

    lr = LogisticRegression(C = best_c, penalty = 'l1')
    lr.fit(X_train,y_train.values.ravel())
    y_pred = lr.predict(X_test.values)

    cnf_matrix = confusion_matrix(y_test,y_pred)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

    class_names = [0,1]
    plt.figure()
    create_confusion_matrix(cnf_matrix,
                            classes=class_names,
                            title='Confusion matrix')
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv("../Downloads/creditcard.csv")
    df = df.rename(columns={'Class': 'fraud'})
    features = df.columns[1:29]
    
    analyze_features(features,df)

    '''
    Drop features from dataframe that have similar distributions among transactions that are fraud and
    not fraud.  Similar distributions indicate that these features will not be effective in determining classification.
    '''
    df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1, inplace = True)
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].reshape(-1, 1))
    df.drop(['Time','Amount'],axis=1,inplace=True)
    df = feature_engineering(df)

    X_train, X_test, y_train, y_test, X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = create_split(df)

    best_c = printing_Kfold_scores(X_train_undersample,y_train_undersample)

    plot_confusion_matrix(X_train_undersample,y_train_undersample,X_test_undersample,y_test_undersample,best_c)
    plot_confusion_matrix(X_train_undersample,y_train_undersample,X_test,y_test,best_c)
    