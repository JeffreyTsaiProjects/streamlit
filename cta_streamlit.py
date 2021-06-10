import os
import time as time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

import streamlit as st


class Data:
    coord_d = {'LincolnSquare':[41.97,-87.69],
                 'LoganSquare':[41.928333,-87.706667],
                 'WickerPark':[41.9075,-87.676944]} 
    
    
    def __init__(self,file_path,fname_processed):
        self.file_path = file_path
        self.fname_processed = fname_processed
        self.df = None
        self.stationlist = []
        
#     @st.cache    
    def cache_data(self):
        '''
        cache processed data for pd.read_csv
        '''
        self.df = pd.read_csv(self.file_path+self.fname_processed,parse_dates=['date'],date_parser=pd.to_datetime)
        
    @staticmethod
    def filter_dates(df,date0=None,date1=None,verbose=False):
        '''
        filter dataframe by dates
        df       : data source, DataFrame
        date0    : start date inclusive in format YYYYMMDD, None or str
        date1    : end date inclusive in format YYYYMMDD, None or str
        verbose  : print rows dropped details, bool
        returns  : DataFrame
        '''
        n0 = len(df)
        if date0 and date1:
            result_df =  df[(df['date']>=pd.to_datetime(date0))&
                         (df['date']<=pd.to_datetime(date1))]
            if verbose:
                n_dropped = n0-len(result_df)
                print(f'filter_dates between {date0},{date1} dropped {n_dropped} rows')
            return result_df
        elif date0 and not date1:
            result_df =  df[df['date']>=pd.to_datetime(date0)]
            n_dropped = n0-len(result_df)
            if verbose:
                print(f'filter_dates greater than or equal {date0} dropped {n_dropped} rows')
            return result_df
        elif date1 and not date0:
            result_df = df[df['date']<=pd.to_datetime(date1)]
            n_dropped = n0-len(result_df)
            if verbose:
                print(f'filter_dates less than or equal {date1} dropped {n_dropped} rows')
            return result_df
        else: pass    
        
        
    def select_stations(self,stationlist,datefilter=None,catfilter=None,
                       zscore=None):
        '''
        select station(s) available stations in dataset
        stationlist: list of stationname values, str
        datefilter : start and end dates inclusive, None or list of None or str
        catfilter  : cat feature name to cat value kwarg, None or dict 
        zscore     : zscore_filter params, score grp type/scoreval/upper bool, list
        returns    : DataFrame
        '''
#         df = self.df[self.df['stationname']==station]
        df = self.df[self.df['stationname'].isin(stationlist)]
        if catfilter:
            catcol,colval = list(catfilter.items())[0]
            df = df[df[catcol]==colval]
        if zscore:
            zscore_grp,zscore_val,upper_only = zscore[0],zscore[1],zscore[2]
            df = self.zscore_filter(df,zscore_grp,zscore_val,upper_only,verbose=True)
        if datefilter:    
            df = self.filter_dates(df,datefilter[0],datefilter[1])
        return df    
        
        
#     def station_stats(self,station,datefilter,catfilter=None,grp=None):
#         '''
#         compute station rides descriptive statistics and dist plots
#         '''
#         df = self.select_stations(station,datefilter=datefilter,catfilter=catfilter)
#         if grp:
#             grplist = ['date']+[grp]
#         else:
#             grplist = ['date']
#         g = df.groupby(grplist,as_index=False)
#         aggs_d = {'rides':['count','mean','std','min','median','max']}
#         g_df = g.agg(aggs_d)
#         g_df.columns = g_df.columns.map('_'.join)
#         g_df.rename(columns={'stationname_':'stationname'},inplace=True)
        
#         return g_df
        
    def station_stats(self):
        '''
        compute station point estimates table
        '''
        g = self.df.groupby(['stationname','linename'],as_index=False)
        # merge linename stats
        iqr_ratio = lambda x: x.quantile(.75)/x.quantile(.25)
        q1 = lambda x: x.quantile(.25)
        q3 = lambda x: x.quantile(.75) 
        agg_dict = {'rides':['mean','std','count',q1,iqr_ratio,'median',q3,'max'],
                   'longitude':'last',
                   'latitude':'last',
                    }
        for hood in d.coord_d:
            agg_dict[f'distance_from_{hood}'] = 'last'    

        g_df = g.agg(agg_dict)
        g_df.columns = g_df.columns.map('_'.join)
        g_df.rename(columns={'stationname_':'stationname','rides_<lambda_0>':'rides_q3/q1',
                            'rides_<lambda_1>':'rides_q1','rides_<lambda_2>':'rides_q3',
                            'linename_':'linename'},inplace=True)
        g_df['rides_mean/std'] = g_df['rides_mean']/g_df['rides_std']  # ridership consistencu
        g_df['rides_mean/median'] = g_df['rides_mean'] / g_df['rides_median']  # skew

        return g_df 
        
class EDA:
    def __init__(self,data):
        self.data = data
        
        
    @staticmethod    
    def distributionplot(df,x,y,hue=None,rot=None,figsz=(12,6)):
        '''
        vizualize data distribution
        df       : data source, DataFrame
        x        : x-axis feature for boxplot, str
        y        : x-axis for kde plot and y-axis feature for boxplot, str
        hue      : category to separate in plot, str
        rot      : pyplot xticks rotation angl, float
        figsz    : pyplot figure dim, tuple of int or None
        '''
        cat_order=None
        if x:
            g = df.groupby(x,as_index=False)
            g_df = g[y].mean().sort_values(by=y)
            cat_order = g_df[x].values
        
        with plt.style.context('seaborn'):
            fig,(ax0,ax1) = plt.subplots(1,2,figsize=figsz)
            sns.histplot(data=df,x=y,hue=hue,kde=True,ax=ax0) 
            sns.boxplot(data=df,x=hue,y=y,hue=None,order=cat_order,ax=ax1)
            plt.xticks(rotation=rot)
            return ax1.get_figure()

    @staticmethod    
    def displots(df,x,y,hue=None,rot=None,figsz=(12,6)):
        '''
        vizualize bimodal data distribution with facetgrid displot
        df       : data source, DataFrame
        x        : x-axis feature for boxplot, str
        y        : x-axis for kde plot and y-axis feature for boxplot, str
        hue      : category to separate in plot, str
        rot      : pyplot xticks rotation angl, float
        figsz    : pyplot figure dim, tuple of int or None
        '''
        fg = sns.FacetGrid(data=df)
        fg.map(sns.displot, x='rides', y=hue)
        return fg.fig
        
    def heatmap_xy(self,station,x,y,stat,datefilter=None,figsz=(8,4)):
        '''
        compute pivot table with x, y and viz a heatmap 
        station    : stationame, str
        x          : pivot table x-axis feature, str
        y          : pivot table y-axis feature, str
        stat       : statistic to compute in groupby agg, str
        datefilter : dates to filter in select_station, list of str
        figsz      : pyplot figure size, tuple of int
        returns    : pyplot figure
        '''
        df = self.data.select_station(station,datefilter=datefilter)
        fig,ax=plt.subplots(figsize=figsz)
        g = df.groupby([x,y],as_index=False)
        g_df = g.agg({'rides':stat})
        pvt = g_df.pivot(y,x,'rides').sort_index(axis=1, level=1)
#         display(pvt)
        sns.heatmap(pvt)
        ax.set(title=f'{station} {stat} rides')
        return ax.get_figure()
            
    def cta_map_rides(self,
                      file_name,
                      filtercols=None,
                      stationlist=None,
                      coordlist=None,
                      zscore=['daytype',3,True],
                      size=None,
                      color='r',alphas=[.2,.7],axlims=[-87.95,-87.51,41.695,42.119], 
                      figsz=(14,14),axOnly=False,
                      ):
        '''
        visualize CTA L station mean daily rides, overlayed on CTA map
        file_name  : name of image file to read in working directory, str
        filtercols : pre-filter kwargs column keys to column values, dict
        stationlist: display multiple stations, None or list of str
        coordlist  : override to display coordinates in this list, list of list of floats
        zscore     : zscore_filter params, score grp type/scoreval/filter upper bool, list 
        size       : scatter plot point size, None or int
        color      : pyplot color, str
        alphas     : pyplot alphas, list of flt
        axlims     : pyplot axis limits, None or list of floats
        figsz      : pyplot figure dimension, tuple of flt
        axOnly     : return pyplot axis art only to overlay on existing figure, bool
        returns    : pyplot fig 
        '''
        df = self.data.df.copy()
        if filtercols:
            for tup in filtercols.items():
                col,val = tup[0],tup[1]
                df = df[df[col]==val]
                
        if stationlist:
            df = df[df['stationname'].isin(stationlist)]
            
        grp_df = df.groupby(['stationname','longitude','latitude'],as_index=False).agg({'rides':'mean',
                                                                                             'longitude':'mean',
                                                                                             'latitude':'mean'})
        if coordlist:
            fig,ax = plt.subplots(figsize=figsz)
            for coord in coordlist:
                long,lat = coord[0],coord[1]
                plt.scatter(x=lat,y=long,alpha=alphas[0],
                            s=size,label=None,
                            c=color,
            #                 c=grp_df['rides'],cmap=plt.get_cmap("jet"),
                            )
        else:
            ax = grp_df.plot(x='latitude',y='longitude',kind='scatter',alpha=alphas[0],
                        s=grp_df['rides']/5,label=None,
                        c=color,
        #                 c=grp_df['rides'],cmap=plt.get_cmap("jet"),
                        figsize=figsz)
            
#         path = os.path.join('/home','jst2136','DataProjects')
#         fname = 'cta_Lmap.png'
        cta_img=mpimg.imread(os.path.join(self.data.file_path,file_name))
        plt.imshow(cta_img,extent=axlims,alpha=alphas[1])
        return ax.get_figure()
        
    def kmeans_elbow_plot(self,x,y,n=None,figsz=(5,5)):
        '''
        compute kMeans cluster elbow plot of x=k, y=KMeans(k) inertia_
        where inertia_ is the sum of xi from their centroids
        data     : 2-minensional x,y data source, DataFrame
        x        : x-axis feature, str
        y        : y-axis feature, str
        n        : number of clusters to fit for elbow plot, None or int
        figsz    : pyplot figure dimensions, tuple of float
        returns  : pyplot figure
        '''
        if not n:
            n = int(input('How many KMeans algos to fit for elbow plot?: '))

        # mean rides    
#         g = self.data.df[['stationname',x,y]].groupby(['stationname'],as_index=True)
#         g_df = g.mean() 
#         g_df = g_df.dropna().sort_values('rides')
        
#         g_df = self.station_stats_df #.sort_values('rides_mean')
        df = self.data.station_stats_df[[x,y]]
        # compute interias 
        sse = []
        seq = [i+1 for i in range(n)]
        for i in seq:
            pipe = make_pipeline(StandardScaler(),KMeans(n_clusters=i,init='k-means++'))
            pipe.fit(df)
            sse.append(pipe.named_steps['kmeans'].inertia_)

        sns.set_style('darkgrid')
        fig,ax = plt.subplots(figsize=figsz)

        kmeans_df = pd.DataFrame(data={'k':seq,'inertia':sse})

        plt.plot('k','inertia',data=kmeans_df,marker='o',label='inertia')
        ax.set(title=f'KMeans Elbow Plot for {x}, {y}',
               xlabel='k clusters',
               ylabel='inertia')
#         plt.pause(.2) # display active figure before the pause


    def cluster_kmeans(self,x,y,k,size=100,cmap='jet',alpha=.7,figsz=(5,5)):
        '''
        determine optimal number of cluster by elbow plot method (min ssd to obs centroids)
        and perform k_means clustering 
        x         : x-axis feature to include in stationname groupby, str
        y         : y-axis response, str
        k         : number of clusters to use in KMeansalgo, int
        size      : scatterplot point size,int
        cmap      : color map, str
        alpha     : scatterplot point transparency, flt
        figsz     : pyplot figure dimensions, tuple of float
        returns   : pyplot fig
        '''

        df = self.data.station_stats_df.set_index('stationname',drop=True)
        df = df[[x,y]]
        pipe=make_pipeline(StandardScaler(),KMeans(n_clusters=k))
        pipe.fit(df)
        labels=pipe.predict(df)

        sns.set_style('darkgrid')
        fig,ax = plt.subplots(figsize=figsz)
        sns.scatterplot(x=x,y=y,data=df,s=size,palette=cmap,alpha=alpha,hue=labels)
        stns = [x for x in df.index]
        for label,x_,y_ in zip(stns, df[x],df[y]):
            plt.annotate(label, xy = (x_, y_), xytext = (-5, 5),
                         textcoords ='offset points',ha='right',va='bottom',
                         alpha=.6,size=8)
        ax.set(title=f'CTA L Station Mean Ridership')
        plt.pause(.2)
        df['cluster'] = labels
        return df.sort_values(by=['cluster',x],ascending=False)
    
        
# path = '/home/jst2136/DataProjects/Streamlit/streamlit' 
path = '~/DataProjects/Streamlit/streamlit' 
fname_processed = 'cta_data_processed.csv'
d = Data(path,fname_processed)
e = EDA(d)

d.cache_data()         

d.stationlist = list(set(d.df['stationname']))
d.stationlist.sort() 

st.title('Analysis of CTA L Ridership')


min_dt = datetime(2001,1,1)
max_dt = datetime(2021,2,28)
date0_selected = st.sidebar.date_input('Which start date?',value=min_dt,min_value=min_dt,max_value=max_dt)
date1_selected = st.sidebar.date_input('Which end date?',value=max_dt,min_value=min_dt,max_value=max_dt)

date0_str = date0_selected.strftime('%Y%m%d')
date1_str = date1_selected.strftime('%Y%m%d')

dt0_str = date0_selected.strftime('%m-%d-%Y')
dt1_str = date1_selected.strftime('%m-%d-%Y')

st.write(f'Dates selected: {dt0_str} - {dt1_str}')

catlist = ['daytype','linename','season','month','dayname','pandemic']

stations_selected = st.sidebar.multiselect('Filter by station(s)?',d.stationlist,default=None)
if len(stations_selected)>1:
    s_str = 'Stations'
elif len(stations_selected) <1:    
    s_str = 'All stations'
else:
    s_str = 'Station'
    
st.write(f"{s_str} selected: {', '.join(stations_selected)}")

catselected = st.sidebar.radio('Apply hue?',[None]+catlist)

if stations_selected:
    df = d.df[d.df['stationname'].isin(stations_selected)]
else:
    df = d.df.copy()
df = df[(df['date']>=pd.to_datetime(dt0_str))&(df['date']<=pd.to_datetime(dt1_str))]    
# if catselected: 
fig = e.distributionplot(df,catselected,'rides',hue=catselected,rot=90,figsz=(14,5))
# fig = e.displots(data=df,x='rides',hue=catselected,rot=None,figsz=(12,6))
st.pyplot(fig)

show_map_bool = st.sidebar.checkbox(f'Show {s_str.lower()} on CTA map?')
if show_map_bool:
    fig = e.cta_map_rides(
                      'cta_Lmap.png',
                      filtercols=None,
                      stationlist=stations_selected,
                      coordlist=None,
                      zscore=['daytype',3,True],
                      size=None,
                      color='r',alphas=[.5,1],axlims=[-87.95,-87.51,41.695,42.119], 
                      figsz=(14,14),axOnly=False,
                      )
    st.pyplot(fig)    
         
         
    