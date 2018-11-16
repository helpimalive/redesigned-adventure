from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


import pandas as pd
import pickle as pickle
import numpy as np
import pypyodbc as odbc
from scipy import stats

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures    
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
    normalize(data,verbose)    

"_______________________BUILDING A MODEL TO PREDICT NEXT MONTHS SECTOR RETURNS_______________________"

def converge_ml(trials,lookback,train_q,test_q):

    stop = test_q
    start = min([train_q])
    preds = make_preds(start,stop,lookback)

    for tial in range(1,trials):
        pred2 = make_preds(start,stop,lookback)
        preds = pd.concat([pred2,preds]) 

    
    final_df = preds.groupby(['PMSector1']).mean().reset_index()

    return(final_df)

"____________CALCULATE TOTAL RETURNS OF BUY SELL SPREAD BASED ON ABOVE MODEL_________"

def calc_trt(trials,picks,lookback,train_q = None,test_q=None):

    buy_sects=pd.DataFrame()
    sell_sects=pd.DataFrame()

    #    predicted total return values
    df_preds = converge_ml(trials,lookback,train_q,test_q)

    #   sort by our predicted total returns descending (highest to lowest)
    df_preds = df_preds.sort_values(by='pred_trt',ascending=False)
    buys  = df_preds.iloc[0:picks]['actual_trt']
    buys  = np.mean(buys)
    buy_recs = df_preds.iloc[0:picks][['PMSector1','actual_trt']]
    sells = df_preds.iloc[len(df_preds)-picks:len(df_preds)]['actual_trt']
    sells = np.mean(sells)
    sell_recs = df_preds.iloc[len(df_preds)-picks:len(df_preds)][['PMSector1','actual_trt']]
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
    if not start:
        start = '2015-01-01'
    if not stop:
        stop = quarters[quarters.index(pd.to_datetime('2015-01-01'))+lookback]

    q=1
    while pd.to_datetime(stop)<max(quarters):
        ml = calc_trt(num_rounds,num_picks,lookback,start,stop)
        results.append(ml)
        print("testing "+str(stop))
        print("results after "+str(q)+" months="+str(np.prod([x['spread']+1 for x in results])))
        print("("+str(np.prod([x['spread']+1 for x in results])**(1/(q)))+") per month")
        start = quarters[quarters.index(pd.to_datetime(start))+1]
        stop = quarters[quarters.index(pd.to_datetime(stop))+1]
        q+=1


    spread = [x['spread'] for x in results]
    spread = pd.DataFrame(spread)
    title = 'spread_results_hptesting'+str(lookback)+'.csv'
    spread.to_csv(title)
    buy_recs = pd.concat([x['buy_recs'] for x in results])
    # buy_recs.to_csv('buy_results_hptesting.csv')
    sell_recs = pd.concat([x['sell_recs'] for x in results])
    # sell_recs.to_csv('sell_results_hptesting.csv')
    full_ranks = pd.concat([x['ranks'] for x in results])
    # full_ranks.to_csv('full_rank_results_hptesting.csv')

    # fr = pd.read_csv('full_rank_results_hptesting.csv')
    # fr['relative_trt']=fr['actual_trt'].groupby(fr['date_bom']).transform('mean')
    # fr['relative_trt']=fr['actual_trt']-fr['relative_trt']
    # fr['ranks']=fr.groupby(['date_bom'])['pred_trt'].transform(lambda x: pd.qcut(x,3,labels=range(1,4)))
    # print(fr.groupby(['ranks'])['relative_trt'].mean())
    # results after 45 months=1.5877778612606486


def make_preds(start,stop,lookback):
    X = pd.read_pickle('Xdatanormalized.p')
    y = pd.read_pickle('Ydatanormalized.p')
    X = X.merge(y,how='left',on=['PMSector1','date_bom'])
    X = X.dropna(subset=['trt'])

    test_X = X[X['date_bom']==stop]
    test_sectors = test_X['PMSector1']
    test_y = test_X['trt']
    test_X = test_X.drop(['trt','date_bom','PMSector1'],axis=1)

    X = X[(abs(X['trt'])<0.20)]
    y = X['trt']
    y = np.log(1+y)
    X = X.drop(['trt'],axis=1)
    X.reset_index(inplace=True)

    X_date_bom = pd.DataFrame(X['date_bom'],columns=['date_bom'])
    X=X[X.columns.difference(['date_bom','PMSector1','index'])]
    index_vals = X_date_bom

    pca = PCA()
    selection = SelectKBest()
    poly = PolynomialFeatures()
    combined_features = FeatureUnion([("pca", pca),
                                     ("univ_select", selection),
                                     ("poly",poly)])

    ada = AdaBoostRegressor()
    rfr = RandomForestRegressor()
    imputer = Imputer()

    pipeline = Pipeline(steps=[
                            ('imputer',imputer),
                            ('features',combined_features),
                            ('regressor',ada)])
    param_grid =[{
    'features__pca__n_components':[5,8],
    'features__univ_select__k':[2],
    'features__poly__degree':[2],
    'regressor':[rfr],
    'regressor__n_estimators':[91,121,151],
    'regressor__criterion':['mse'], 
    'regressor__max_depth':[5,9],
    'regressor__min_samples_split':[2]
    },
    {
    'features__pca__n_components':[5,8],
    'features__univ_select__k':[2],
    'features__poly__degree':[2],
    'regressor':[ada],
    'regressor__n_estimators':[21,51,81]
    }
    ]

    tscv = custom_timeseries_cv(index_vals = X_date_bom,lookback=lookback,start=start,stop=stop)

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=tscv, verbose=1)
    grid_search.fit(X,y)
    print(grid_search.best_score_)
    print(grid_search.best_params_)
    preds = grid_search.predict(test_X)
    final = pd.concat([pd.DataFrame(preds.reshape(-1)),test_y.reset_index(drop=True)],axis=1,ignore_index=True)
    final = pd.concat([final,test_sectors.reset_index(drop=True)],axis=1,ignore_index=True)
    final.columns = ['pred_trt','actual_trt','PMSector1']
    return(final)

class custom_timeseries_cv:
    def __init__(self, index_vals, n_splits=3, lookback = 12, start=None,stop=None):
        self.n_splits = n_splits
        self.lookback = lookback
        self.index_vals = index_vals
        self.start = start
        self.stop = stop
        

    def split(self, X, y, groups=None):
        self.X = X
        self.y = y
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

        for q in np.arange(end,begin+self.lookback-1,-1):
            test_q = pers[q]
            train_q = pers[q-self.lookback]

            # train = self.index_vals[(self.index_vals['date_bom']<test_q) & 
            #                             (self.index_vals['date_bom']>train_q)].index.values.astype(int)
            # test = self.index_vals[(self.index_vals['date_bom']==test_q)].index.values.astype(int)
            train = self.index_vals[(self.index_vals['date_bom']<test_q) & 
                                        (self.index_vals['date_bom']>train_q)].index.values.astype(int)
            test = self.index_vals[(self.index_vals['date_bom']==test_q)].index.values.astype(int)
            # train = self.X.iloc[[1,2,3,4,5,6,7]].index.values.astype(int)
            # test = self.X.iloc[[1,2,3,4,5,6,7]].index.values.astype(int)
            yield train,test          

    def vals(self, groups=None):
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
            train = self.X[(self.index_vals['date_bom']<test_q) & 
                                        (self.index_vals['date_bom']>train_q)].index.values.astype(int)
            test = self.X[(self.index_vals['date_bom']==test_q)].index.values.astype(int)
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

        self.n_splits = len(np.arange(end,begin+self.lookback-1,-1))
        return self.n_splits
