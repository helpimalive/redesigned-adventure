import pandas as pd
import pickle as pickle
import numpy as np
import pypyodbc as odbc
from scipy import stats


from sklearn.tree import _tree
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score

import os
os.chdir('S:\\MattL\\Machine Learning\\Sector Picker\\code\\big sectors')

"__________________COMPILE ALL RAW DATA__________________________________________________"

def query(verbose=False):

    querytext = "select\
        comp_id,\
        date_bom,\
        PMSector1, \
        EBITDAMultiple1,\
        dividendYield,\
        capExReserve_primary,\
        equityMarketCapUSD,\
        nominalCapRate_primary,\
        economicCapRate_primary,\
        impliedNominalCapRate,\
        fullyLoadedImpliedCapRate_SSNOIGrowth_Pct,\
        SSNOIGrowth1,\
        SSNOIGrowth2,\
        SSNOIGrowth3,\
        SSNOIGrowth4,\
        AFFOGrowth_forward12M,\
        pctAFFOChange1,\
        pctAFFOChange2,\
        pctAFFOChange3,\
        pctAFFOChange4,\
        AFFOYield0,\
        AFFOYield1,\
        AFFOYield2,\
        AFFOYield3,\
        AFFOYield4\
        FROM PET3.dbo.xlPET_Data_HISTORICAL_MONTHLY \
            where date_bom >= '2010-01-01'"

    connection = odbc.connect('Driver={SQL Server};''Server=GS-OLDDB;''Database=PET3;')
    df1 = pd.read_sql(querytext,connection)

    if verbose:
        df1.to_csv('query_data.csv')
    
    return df1

"CONVERTING THE INDIVIDUAL REIT RESULTS TO SECTOR AVERAGES"
def turn_into_avg(df, verbose=False):

    data = df
    # data.index = data.date_bom
    data = data.groupby(['pmsector1','date_bom']).mean()
    data.reset_index(inplace=True)
    data.drop(['comp_id'],axis=1,inplace=True)

    return data

"THIS PREPS A SECTOR-LEVEL TOTAL RETURN FILE"
def trt(df, verbose = False):
    query_text = "select\
        a.date_bom\
        ,a.PMSector1\
        ,a.comp_id\
        ,c.date\
        ,avg(a.equityMarketCapUSD) as avg_mktcap\
        ,EXP(SUM(LOG(NULLIF(1+b.trt,1))))-1 as 'trt'\
        from xlPET_Data_HISTORICAL_MONTHLY a\
        left join  xlPET_Data_HISTORICAL_MONTHLY c on a.date_bom = dateadd(month,-1,c.date_bom) and a.comp_id = c.comp_id\
        left join tblPET_TRTSNL b on a.comp_id = b.comp_id and b.date between\
            dateadd(day,1,a.date) and c.date\
        where a.date>'2005-01-01' and b.trt is not null and b.trt>-1\
        group by a.comp_id,a.date_bom,c.date,a.PMSector1"

    connection = odbc.connect('Driver={SQL Server};''Server=GS-OLDDB;''Database=PET3;')
    df1 = pd.read_sql(query_text,connection)
    df1 = df1[pd.notnull(df1['avg_mktcap'])]

    df1['trt_x_mktcap'] = df1.trt*df1.avg_mktcap
    df2 = df1.groupby(['date_bom','pmsector1']).sum()
    df2.reset_index(inplace=True)
    df2.trt = df2.trt_x_mktcap/df2.avg_mktcap
    df2 = df2[['pmsector1','date_bom','trt']]
    
    return df2

"CREATING MOMENTUM"
def momentum_and_vol(df, verbose=False): 
    df1 = df
    df1 = df1.sort_values(by=['pmsector1','date_bom'])
    dates = pd.DataFrame(list(set(df1['date_bom'])),columns=['date_bom'])
    dates = dates.sort_values(by='date_bom')
    df1 = df1.pivot(index='date_bom',columns='pmsector1')
    df1.reset_index(inplace=True)
    df1.columns = df1.columns.droplevel()
    df1 = df1.drop(df1.columns[0],axis=1)
    momentum = df1.rolling(18).mean().shift(1).diff(1)
    df2 = pd.concat([dates.reset_index(drop=True),momentum],axis=1)
    df3 = pd.melt(df2,id_vars=['date_bom'])
    df3.columns = ['date_bom','pmsector1','momentum']
    df4 = df3

    vol = df1.rolling(18).std().shift(1)
    df2 = pd.concat([dates.reset_index(drop=True),vol],axis=1)
    df3 = pd.melt(df2,id_vars=['date_bom'])
    df3.columns = ['date_bom','pmsector1','vol']

    df4 = pd.merge(df3,df4,right_on=['date_bom','pmsector1'],left_on=['date_bom','pmsector1'])
    
    return(df4)

# "NORMALIZING COMPILED DATA"
def normalize(data,verbose=False):

    m_date = max(data.date_bom)
    data = data[ (np.isfinite(data['trt'])) | (data['date_bom']==m_date)]
    
#   Y values
    y = data[['pmsector1','date_bom','trt']]
    y = y.rename(columns ={'pmsector1':'PMSector1'})
            
#   drop Y values from X data
    data.drop(['trt'],axis=1,inplace=True)

#   STANDARDIZATION

    df = data.groupby(['pmsector1','date_bom']).apply(lambda x: x.fillna(x.mean()))
    df = df.drop(['pmsector1','date_bom'],axis=1)
    df = df.reset_index()
    df = df.drop('level_2',axis=1)
    cols = [x for x in list(df.columns) if x not in ['date_bom','pmsector1']]

    df = df.fillna(0)
    PMSector1 = df.pmsector1
    date_bom  = df.date_bom 
    df = df.groupby(['date_bom']).transform(lambda x: (x - x.mean()) / x.std())
    df['PMSector1']=PMSector1
    df['date_bom'] =date_bom
    
    pickle.dump(df,open('Xdatanormalized.p','wb'))
    pickle.dump(y,open('Ydatanormalized.p','wb'))

"THIS EXECUTES ALL OF THE ABOVE PROCESSES AND RETURNS ONE DATAFRAME"
"THAT CONTAINS AVERAGED METRICS, BY SECTOR, AND AVERAGE TRT BY SECTOR"
def compile_data(verbose=False):
    df = query(verbose)
    df = turn_into_avg(df, verbose)
    df_trt = trt(verbose)
    data = df.merge(df_trt,how='left',on=['pmsector1','date_bom'])
    df_m_v = momentum_and_vol(df_trt, verbose)
    data = data.merge(df_m_v,how='left',on=['pmsector1','date_bom'])
    
    if verbose:
        data.to_csv('input_data.csv')
    # normalize(data,verbose)    

"_______________________BUILDING A MODEL TO PREDICT NEXT MONTHS SECTOR RETURNS_______________________"

def converge_ml(trials,train_q=None,test_q=None):

    x = pd.read_pickle('Xdatanormalized.p')
    y = pd.read_pickle('Ydatanormalized.p')

    #   remove sectors without lookback_months worth of training data
    # full_data_sectors = list(x.PMSector1[(x.date_bom.isin(train_q))])
    # full_data_sectors = dict((x,full_data_sectors.count(x)) for x in set(full_data_sectors))
    # full_data_sectors = [x for x in full_data_sectors if full_data_sectors[x]==len(train_q)]

    x = x.merge(y,how='left',on=['PMSector1','date_bom'])
    # x = x[x.PMSector1.isin(full_data_sectors)]

    holder_df=pd.DataFrame()

    dates = list(set(x.date_bom))

    train_x = x[x.date_bom.isin(train_q)]

    ## Elimination of rogue data and scaling the TRT
    ## NOT SURE ABOUT THESE
    x = x[x['PMSector1']!='Hotel']
    train_x = train_x.dropna(axis='columns')
    train_x = train_x[(abs(train_x['trt'])<0.20)]

    train_y = train_x['trt']
    train_y = np.log(1+train_y)
    train_x = train_x.drop(['trt'],axis=1)

    test_x  = x[x.date_bom.isin([test_q])]
    test_x = test_x[list(train_x.columns)]
    test_x = test_x.dropna(axis='columns')

    train_x = train_x[list(test_x.columns)]

    ## Eliminating any unwanted variables
    l = list(train_x.columns)
    l = [a for a  in l if a not in ['trt','date_bom','PMSector1']]
    summary_lcv = summary_rfr = summary_gbr = l

    ## Preprocessing
    sectors = test_x['PMSector1']
    cols = [a for a  in list(test_x.columns) if a not in ['trt'
                                                            ,'date_bom'
                                                            ,'PMSector1'
                                                            ,'pctaffochange1'
                                                            ,'pctaffochange2'
                                                            ,'pctaffochange3'
                                                            ,'pctaffochange4'
                                                            
                                                            ]]
    train_x = train_x[cols]
    test_x = test_x[cols]

    ##PCA
    from sklearn.decomposition import PCA
    n_components = min([len(cols),7])
    pca = PCA(n_components=n_components)

    ##Add back some originals
    from sklearn.feature_selection import SelectKBest
    selection = SelectKBest(k=2)

    ##Add some interaction effects and polynomials
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(3)

    from sklearn.pipeline import FeatureUnion
    combined_features = FeatureUnion([('pca',pca),('univ_select',selection),('poly',poly)])

    processed_train_x = combined_features.fit(train_x, train_y).transform(train_x)
    processed_test_x = combined_features.fit(train_x, train_y).transform(test_x)

    for run in range(0,trials):
       
    # #       Random Forest Regressor ORIGINAL
        regr = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=9,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=1e-07, min_samples_leaf=2,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=91, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)
    #     # regr = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=12,
    #     #    max_features='auto', max_leaf_nodes=None,
    #     #    min_impurity_decrease=1e-07, min_samples_leaf=4,
    #     #    min_samples_split=2, min_weight_fraction_leaf=0.0,
    #     #    n_estimators=41, n_jobs=1, oob_score=False, random_state=None,
    #     #    verbose=0, warm_start=False)

    #     # x_train_rfr=train_x[summary_rfr]
    #     # x_test_rfr =test_x[summary_rfr]

        regr.fit(processed_train_x,train_y)
        rfr_preds= regr.predict(processed_test_x)

    #     #        Gradient Boosting Regressor
        # gbr = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
        #      learning_rate=0.1, loss='lad', max_depth=3, max_features=None,
        #      max_leaf_nodes=None, min_impurity_decrease=1e-07,
        #      min_samples_leaf=3, min_samples_split=2,
        #      min_weight_fraction_leaf=0.0, n_estimators=300,
        #      presort='auto', random_state=None, subsample=1.0, verbose=0,
        #      warm_start=False)

    #     # x_train_gbr=train_x[summary_gbr]
    #     # x_test_gbr =test_x[summary_gbr]        

        # gbr.fit(processed_train_x,train_y)
        # gbr_preds=gbr.predict(processed_test_x)

    #     #       Bagging Regressor
    #     lcv = BaggingRegressor(base_estimator=None, bootstrap=True, bootstrap_features=True,
    #              max_features=2, max_samples=1.0, n_estimators=100, n_jobs=1,
    #              oob_score=False, random_state=None, verbose=0, warm_start=False)

    #     # x_train_lcv = train_x[summary_lcv]
    #     # x_test_lcv  = test_x[summary_lcv]

    #     lcv.fit(processed_train_x,train_y)
    #     lcv_preds=lcv.predict(processed_test_x)

        #       ADA Regressor
        ada = AdaBoostRegressor()
        ada.fit(processed_train_x,train_y)
        ada_preds=ada.predict(processed_test_x)

        #       Combining and averaging all results
        temp_df= pd.DataFrame({
                                # 'lcv':[a for a in lcv_preds]
                               'rfr':[a for a in rfr_preds]
                               # ,'gbr':[a for a in gbr_preds]
                               ,'ada':[a for a in ada_preds]
                               })        

        #       Appending the results of this trial to a dataframe
        holder_df[run]=temp_df.mean(axis=1)

    #   Averaging all predictions        
    final_df = pd.DataFrame(holder_df.mean(axis=1),columns=['pred_trt'])
    final_df = final_df.reset_index(drop=True)
    final_df['PMSector1'] = list(sectors)
    final_df['date_bom'] = test_q
    return(final_df)

"____________CALCULATE TOTAL RETURNS OF BUY SELL SPREAD BASED ON ABOVE MODEL_________"

def calc_trt(trials,picks,train_q = None,test_q=None):

    x = pd.read_pickle('Xdatanormalized.p')
    y = pd.read_pickle('Ydatanormalized.p')

    buy_sects=pd.DataFrame()
    sell_sects=pd.DataFrame()

    #    predicted total return values
    df_preds = converge_ml(trials,train_q,test_q)


    #    actual total return values
    y=pd.DataFrame(y)
    y.columns = ['PMSector1','date_bom','actual_trt']

    #    combining predicted total returns and actual total returns on their indexes
    df_preds = df_preds.merge(y,how='left',on=['PMSector1','date_bom'])

    #   combining the fields sector and date_bom with the above-created predicted and actual trt DF    
    fields = x[['PMSector1','date_bom']]
    df_preds = df_preds.merge(fields,how='left',on=['PMSector1','date_bom'])

    #   sort by our predicted total returns descending (highest to lowest)
    df_preds = df_preds.sort_values(by='pred_trt',ascending=False)
    buys  = df_preds.iloc[0:picks]['actual_trt']
    buys  = np.mean(buys)
    buy_recs = df_preds.iloc[0:picks][['date_bom','PMSector1','actual_trt']]
    sells = df_preds.iloc[len(df_preds)-picks:len(df_preds)]['actual_trt']
    sells = np.mean(sells)
    sell_recs = df_preds.iloc[len(df_preds)-picks:len(df_preds)][['date_bom','PMSector1','actual_trt']]
    spread = buys-sells    
    # print(df_preds)
    print("BUY")
    print(buy_recs)
    print("SELL")
    print(sell_recs)

    ret ={  "spread":spread,
            "ranks": df_preds,
            "buy_recs":buy_recs,
            "sell_recs":sell_recs
            }
    print(df_preds)
    return(ret)

def forward_rounds(num_rounds,num_picks,lookback,start=None,stop=None):
    x = pd.read_pickle('Xdatanormalized.p')
    quarters=list(set(x.date_bom))
    quarters.sort()

    results=list()
    if start:
        start = quarters.index(pd.to_datetime(start))
        stop = quarters.index(pd.to_datetime(stop))
    else:
        start = len(quarters)-1
        stop = lookback+1

    for q in np.arange(start,stop,-1):
        test_q = quarters[q]
        if test_q>=pd.to_datetime('2012-12-01'):
        # if test_q>=pd.to_datetime('2018-01-01'):            
            pass
        else:
            break

        train_q = quarters[q-lookback:q]
        ml = calc_trt(num_rounds,num_picks,train_q,test_q)
        results.append(ml)
        print("testing "+str(test_q))
        print("results after "+str(len(quarters)-q-1)+" months="+str(np.prod([x['spread']+1 for x in results])))
        print("("+str(np.prod([x['spread']+1 for x in results])**(1/(len(quarters)-q-1)))+") per month")


    spread = [x['spread'] for x in results]
    spread = pd.DataFrame(spread)
    title = 'spread_results_hptesting'+str(lookback)+'.csv'
    spread.to_csv(title)
    buy_recs = pd.concat([x['buy_recs'] for x in results])
    buy_recs.to_csv('buy_results_hptesting.csv')
    sell_recs = pd.concat([x['sell_recs'] for x in results])
    sell_recs.to_csv('sell_results_hptesting.csv')
    full_ranks = pd.concat([x['ranks'] for x in results])
    full_ranks.to_csv('full_rank_results_hptesting.csv')

# fr = pd.read_csv('full_rank_results_hptesting.csv')
# fr['relative_trt']=fr['actual_trt'].groupby(fr['date_bom']).transform('mean')
# fr['relative_trt']=fr['actual_trt']-fr['relative_trt']
# fr['ranks']=fr.groupby(['date_bom'])['pred_trt'].transform(lambda x: pd.qcut(x,3,labels=range(1,4)))
# print(fr.groupby(['ranks'])['relative_trt'].mean())
# results after 45 months=1.5877778612606486

from sklearn.model_selection import TimeSeriesSplit
X = pd.read_pickle('Xdatanormalized.p')
y = pd.read_pickle('Ydatanormalized.p')
quarters=list(set(X.date_bom))
quarters.sort()
splits = (len(quarters)-24)
tscv = TimeSeriesSplit(max_train_size = 24,n_splits=splits)
for train_index, test_index in tscv.split(quarters):
    print("TRAIN:", train_index, "TEST:", test_index,quarters[test_index[0]])
    # X_train, X_test = X[train_index], X[test_index]
    # y_train, y_test = y[train_index], y[test_index]


from __future__ import print_function
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures    

X = pd.read_pickle('Xdatanormalized.p')
y = pd.read_pickle('Ydatanormalized.p')
X = X.merge(y,how='left',on=['PMSector1','date_bom'])
X = X.dropna(subset=['trt'])
X = X[(abs(X['trt'])<0.20)]
y = X['trt']
y = np.log(1+y)
X = X.drop(['trt'],axis=1)
X_date_bom = pd.DataFrame(X['date_bom'],columns=['date_bom'])
X=X[X.columns.difference(['date_bom','PMSector1'])]

pipe_vals = list()

## FEATURES
pca = PCA()
selection = SelectKBest()
poly = PolynomialFeatures()
combined_features = FeatureUnion([("pca", pca),
                                 ("univ_select", selection),
                                 ("poly",poly)])

pipe_vals.append(("imputer",Imputer()))
pipe_vals.append(("features",combined_features))

param_grid = dict(features__pca__n_components=[5,7],
                  features__univ_select__k=[2,3],
                  features__poly__degree=[2])

##REGRESSORS
ada = AdaBoostRegressor()
pipe_vals.append(("ada",ada))
param_grid['ada__n_estimators'] = [21,31,41]

## Cross Validation
tscv = custom_timeseries_cv(index_vals = X_date_bom,lookback=12,start='2017-12-01',stop='2018-09-01')

## Pipeline
pipeline = Pipeline([
            ("imputer",Imputer()),
            ("features", combined_features),
            ("ada", ada)
            ])

pipeline = Pipeline(pipe_vals)
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=tscv, verbose=10)
grid_search.fit(X,y)

print(grid_search.best_score_)
print(grid_search.best_params_)
print('total preds = ',sum(grid_search.predict(X)))

class custom_timeseries_cv:
    def __init__(self, index_vals, n_splits=3, lookback = 12, start=None,stop=None):
        self.n_splits = n_splits
        self.lookback = lookback
        self.index_vals = index_vals
        self.start = start
        self.stop = stop

    def split(self, X, y, groups=None):
        pers=list(set(self.index_vals['date_bom']))
        pers.sort()

        if self.start:
            begin = pers.index(pd.to_datetime(self.start))
        else:
            begin = pers.index(pd.to_datetime('2012-12-01'))

        if self.stop:
            end = pers.index(pd.to_datetime(self.stop))
        else:
            end = self.lookback+1

        for q in np.arange(end,begin,-1):
            test_q = pers[q]
            train_q = pers[q-self.lookback]

            # train = self.index_vals[(self.index_vals['date_bom']<test_q) & 
            #                             (self.index_vals['date_bom']>train_q)].index.values.astype(int)
            # test = self.index_vals[(self.index_vals['date_bom']==test_q)].index.values.astype(int)
            train = X[(self.index_vals['date_bom']<test_q) & 
                                        (self.index_vals['date_bom']>train_q)].index.values.astype(int)
            test = X[(self.index_vals['date_bom']==test_q)].index.values.astype(int)

            train = X.iloc[[1,2,3,4,5,6,7]].index
            test = X.iloc[[1,2,3,4,5,6,7]].index
            yield train,test          

    def vals(self, X, y, groups=None):
        pers=list(set(self.index_vals['date_bom']))
        pers.sort()

        if self.start:
            begin = pers.index(pd.to_datetime(self.start))
        else:
            begin = pers.index(pd.to_datetime('2012-12-01'))

        if self.stop:
            end = pers.index(pd.to_datetime(self.stop))
        else:
            end = self.lookback+1

        for q in np.arange(end,begin,-1):
            test_q = pers[q]
            train_q = pers[q-self.lookback]

            # train = self.index_vals[(self.index_vals['date_bom']<test_q) & 
            #                             (self.index_vals['date_bom']>train_q)].index.values.astype(int)
            # test = self.index_vals[(self.index_vals['date_bom']==test_q)].index.values.astype(int)
            train = X[(self.index_vals['date_bom']<test_q) & 
                                        (self.index_vals['date_bom']>train_q)].index
            test = X[(self.index_vals['date_bom']==test_q)].index

            # train = X.iloc[[1,2,3,4,5,6,7]].index
            # test = X.iloc[[1,2,3,4,5,6,7]].index
        return(train,test)

    def get_n_splits(self, X, y, groups=None):
        pers=list(set(self.index_vals['date_bom']))
        pers.sort()

        if self.start:
            begin = pers.index(pd.to_datetime(self.start))
        else:
            begin = pers.index(pd.to_datetime('2012-12-01'))

        if self.stop:
            end = pers.index(pd.to_datetime(self.stop))
        else:
            end = self.lookback+1

        self.n_splits = len(np.arange(end,begin,-1))
        return self.n_splits