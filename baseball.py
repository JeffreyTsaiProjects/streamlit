from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn
import streamlit as st

class Data:
    # league, division, team maps for filtering
    team_league_dict = {'ARI':'NL','ATL':'NL','BAL':'AL','BOS':'AL','CHC':'NL','CHW':'AL','CIN':'NL','CLE':'AL','COL':'NL','DET':'AL',
                            'HOU':'AL','KCR':'AL','LAA':'AL','LAD':'NL','MIA':'NL','MIL':'NL','MIN':'AL','NYM':'NL','NYY':'AL','OAK':'AL',
                            'PHI':'NL','PIT':'NL','SDP':'NL','SEA':'AL','SFG':'NL','STL':'NL','TBR':'AL','TEX':'AL','TOR':'AL','WSN':'NL'}
    div_team_dict = {'E':['BOS','TBR','TOR','NYY','BAL']+['NYM','PHI','ATL','MIA','WSN'],
                         'C':['CHW','CLE','KSR','DET','MIN']+['STL','CHC','MIL','CIN','PIT'],
                         'W':['OAK','HOU','SEA','LAA','TEX']+['SFG','SDP','LAD','ARI','COL']}
    
    def __init__(self):
        self.urldict = {} # url name : url link kwargs
        self.mlb_year = None
        self.season_result_df = None
        self.standings_league_div_dict = {}
        
    def addUrls(self,kwargs):
        '''
        construct url and store to self.urldict
        kwargs    : url name key to url link generic value with date ref, str
        returns   : None
        '''
        for k,v in kwargs.items():
            self.urldict[k] = v
            
    @staticmethod
    def assert_numeric(string):
        '''
        helper function to determine if string contains all numeric values
        string    : string to assert can be cast to float, str
        returns   : bool
        '''
        try: 
            float(string)
            return True
        except ValueError as e:
            print(e)
            return False
            
    def standings(self,year):
        '''
        scrape Baseball Reference team standings for a givven year
        year      : MLB season YYYY format, int or str
        returns   : DataFrame
        '''
        thisYear = datetime.now().year
        
#         year = st.slider('Select a year:',
#                         1900,thisYear,thisYear)
        
        url = self.url_dict['standings'].format(year)
        
        # connect to website
        page = requests.get(url)
        
        #init a beautigul soup object
        bsobj = BeautifulSoup(page.content) 
        
    def BaseballReferenceTeamStatsYear(self,year):
        '''
        scrape Baseball Reference Team Statisics page for a given year
        class_ids : class id in html soup (batting, fielding, pitching,..) list of str
        returns   : DataFrame
        '''
        
        class_id = st.radio('Which type of team statistic?',
                            ['batting','fielding','pitching'])
        
        url = self.urldict[f'{class_id}'].format(year)
        st.write(f'scraping:\n{url}')

        # connect to the website page
        page = requests.get(url)
        # init a BeautifulSoup data object  
        bs_obj = BeautifulSoup(page.content)

        
        class_id_dict = {'batting':'teams_standard_batting',
                         'fielding':'teams_standard_fielding',
                         'pitching':'teams_standard_pitching'}
        
        # the batting table has an id tag teams_standard_batting
        result = bs_obj.find_all(id=class_id_dict[class_id])

        # the table has tr tags
        table = result[0].find_all('tr')
        rowcount = len(table)
#         st.write(rowcount)
        # the header label are the first row
        labels = table[0]
        # clean the newline and empty elements
#         labels = [x.string.replace('\n','') for x in labels if x.string.replace('\n','') != ''] 
        labels = [x.string for x in table[0] if x.string != '\n']
#         st.write(labels)

        # the team stats are in rows 1 to end
        data = table[1:rowcount-3]
        dataDict = defaultdict(list)
        for row in data:
            tmpList = []
            for elem in row:
                value = elem.string

                if self.assert_numeric(value):
                    tmpList.append(float(value))
                else:
                    teamName = value
                    dataDict[teamName] = []  # init list
                    # populate the value list to be assigned to dict key after both loops complete
                    tmpList.append(value)
            # team row is done, so assign key in dict to list value                    
            dataDict[teamName] = tmpList
            
        # construct the dataframe result
        df = pd.DataFrame(columns=labels)
        
        for team in dataDict:
            # get the team data row as DataFrame
            tmp_df = pd.DataFrame(np.array([dataDict[team]]), columns=labels)
            # append the team tmp df to the bottom of the result dataframe 
            df = pd.concat([df,tmp_df] )
        # assign the final result
        self.season_result_df = df.set_index('Tm').copy()
        
        # cast X to numeric
        if self.season_result_df.G.dtype == 'object':
            self.season_result_df = self.season_result_df.apply(lambda x: pd.to_numeric(x,downcast='float') )
            
        show_df = st.checkbox('Display DataFrame?')
        if show_df:
            st.dataframe(d.season_result_df)

    def BaseBallReferenceStandings(self,year):
        '''
        construct league standings tables
        year       : MLB year, int
        returns    : None
        '''
        # league urls
        page_AL = requests.get(self.urldict['standings_AL'].format(year))
        page_NL = requests.get(self.urldict['standings_NL'].format(year))
        # soup objects
        soup_AL = BeautifulSoup(page_AL.content)
        soup_NL = BeautifulSoup(page_NL.content)
#         standings_AL = soup_AL.find_all(id='standings_AL')
#         standings_NL = soup_NL.find_all(id='standings_NL')
        
        standings_d = {}
        
        standings_df = pd.DataFrame()
        
        for league in ['AL','NL']:
            for div in ['E','C','W']:
                if league=='AL':
                    div_standings = soup_AL.find_all(id=f'standings_{div}')
                else:
                    div_standings = soup_NL.find_all(id=f'standings_{div}')
                table = div_standings[0].find_all('tr')
#                 st.write(table)
                tmp_list = []
                for row in table[1:]:
#                     st.write(row)
                    tmp_list = [x.string for x in row]
#                     st.write(tmp_list)
                    tmp_list.append(league)
#                     st.write(league)
                    tmp_list.append(div)
                    team = tmp_list[0]
                    standings_d[team] = tmp_list
                    header = [x.string for x in table[0] if x.string != '\n']
                    header.append('League')                                             
                    header.append('Div')   
                    tmp_df = pd.DataFrame(np.array([standings_d[team]]),columns=header)
                    standings_df = pd.concat([standings_df,tmp_df])
        self.standings_mlb_df = standings_df.sort_values('W-L%',ascending=False).copy()
        
    def division_standings(self):
        '''
        compute standings by league, division
        league    : "AL" or "NL", str
        returns   : None, assigns league,div DataFrame values to keys in self.standings_mlb_dict
        '''
        # store standings in self.standings_league_div_dict 
        for lg in ['AL','NL']:
            for div in ['E','C','W']:
                lg_div_df = self.standings_mlb_df[(self.standings_mlb_df['Div']==div)
                                                 &(self.standings_mlb_df['League']==lg)]
                self.standings_league_div_dict[f'{lg}_{div}'] = lg_div_df.sort_values(by='W-L%',ascending=False)
        # all mlb
#         self.standings_league_div_dict['MLB'] = self.standings_mlb_df
        
        league = st.sidebar.radio('Which league?',['MLB','AL','NL'])
        
        if league=='MLB':
            st.write(f'{self.mlb_year} Standings All MLB Teams')
            st.dataframe(self.standings_mlb_df)
        else:
            division = st.sidebar.radio('Which division?',[None,'East','Central','West'])
            
            if league=='AL':
                if division==None:
                    st.write(f'{self.mlb_year} Standings American League')
                    st.dataframe(self.standings_mlb_df[self.standings_mlb_df['League']=='AL'])
                elif division=='East':
                    st.write(f'{self.mlb_year} Standings American League East')
                    st.dataframe(self.standings_league_div_dict['AL_E'])
                elif division=='Central':
                    st.write(f'{self.mlb_year} Standings American League Central')
                    st.dataframe(self.standings_league_div_dict['AL_C'])
                elif division=='West':
                    st.write(f'{self.mlb_year} Standings American League West')
                    st.dataframe(self.standings_league_div_dict['AL_W'])
            else:
                if division==None:
                    st.write(f'{self.mlb_year} Standings National League')
                    st.dataframe(self.standings_mlb_df[self.standings_mlb_df['League']=='NL'])
                elif division=='East':
                    st.write(f'{self.mlb_year} Standings National League East')
                    st.dataframe(self.standings_league_div_dict['NL_E'])
                elif division=='Central':
                    st.write(f'{self.mlb_year} Standings National League Central')
                    st.dataframe(self.standings_league_div_dict['NL_C'])
                elif division=='West':
                    st.write(f'{self.mlb_year} Standings National League West')
                    st.dataframe(self.standings_league_div_dict['NL_W'])
                
        
                
     
                                                     
                                   
class Stats:
    folds = 5
    
    def __init__(self,data_obj):
        self.data_obj = data_obj
        self.eigvals = None
        self.eigvecs = None
        self.loading_df = None
        self.score_df = None
        
    @staticmethod    
    def center_scale(X):
       '''
       center by subracting xi by mean of vector X 
       and scale by dividing xi by std of vector X
       X          : features, DataFrame
       returns    : DataFrame
       '''
       return (X-np.mean(X)) / np.std(X)
    
    def PCA(self,df,league=None,division=None):
        '''
        compute a principal component analysis on data in df
        df       : data table, DataFraame
        league   : MLB League, None or str
        division : MLB League division, None or str
        returns  : tuple of eigenvalue vector, loading_df, score_df
        '''
        # filter by league
#         if league:
#             tmp_df = df.copy()
#             tmp_df['league'] = Data.team_league_dict[df.index]
#             tmp_df = tmp_df[tmp_df.index==league]
#             st.dataframe(tmp_df)
        #step 1: center and scale the features matrix
        C = self.center_scale(df)
#         st.write(f'n teams:{len(C)}')
#         st.write(f'Centered X shape:{C.shape}')
#         st.dataframe(C)
        
        # step2: compute covariance of transpose of centered features
        # cov is wrt to X features, not team index
        Cov = np.cov(C.T)
#         st.write(f'Covariance(X), Cov shape:{Cov.shape}')
#         st.dataframe(Cov)
        
        # step3 compute the PC loading vectors, directions of greatest variance in X feature space and explained variance (eigenalues)
        eigvals, eigvecs = np.linalg.eig(Cov)
#         st.write(f'Eigenvalues, Eigenvectors shape:{eigvals.shape,eigvecs.shape}')
#         st.write('first and second eigenvalue proportion of total variance:{}'.format(eigvals[:2]/eigvals.sum()) )
        # eigenvectors are PC loading vectors
        loading_colnames = ['L'+str(i) for i in range(1,len(df.columns)+1)]
        #
        loading_df = pd.DataFrame(eigvecs,index=df.columns,columns=loading_colnames).astype(np.float)
#         st.write(f'Loadings vectors shape:{loading_df.shape}')
#         st.dataframe(loading_df)
#         st.write('Top 5 PC loading vectors (directions with largest variance in X feature space):')
#         st.dataframe(loading_df.loc[:,:'L5'])
        
        # step 4: compute score vectors as PCs to reduce dim, by projecting features C onto the loading vectors         
        scores_matrix = loading_df.values.T.dot(C.T) # returns np.array
        colnames_scores_matrix = ['PC'+str(i) for i in range(1,C.shape[1]+1)]
#         st.dataframe(scores_matrix)
#         st.dataframe(scores_matrix.T)
        # scores index is same as original data because X are projected onto a lower dim loading plane 
        score_df = pd.DataFrame(scores_matrix.T,index=C.index,columns=colnames_scores_matrix) 
#         st.write(f'PC matrix, shape {score_df.shape}:')
#         st.dataframe(score_df)
        return eigvals, loading_df, score_df
    
    def PVE(self,eigenvals,figsz=(12,6)):
        '''
        compute percent variance explained by PCA eigevalues
        eigenvals    : vector of eigenvalues from PCA Cov(X) matrix, numpy arr
        returns      : pyplot figure
        '''
        with plt.style.context('seaborn-pastel'):
            fig,ax = plt.subplots(figsize=figsz)
            variance_total = eigenvals.sum()
            # compute proportional variance explained by each PC 
            pve = eigenvals/variance_total
            # compute xticks
            xs = [str(i) for i in range(1,len(eigenvals)+1)]
            plt.plot(xs,pve,label='Pct Variance Explained')
            ax.set(title='Percent Variance Explained by ith Principal Component',
                   xlabel='PC',ylabel='Percent Variance')     
            # compute cumulative variance explained
            pve_cumulative = np.cumsum(pve)
            ax.plot(xs,pve_cumulative,label='Cumulative Variance Explained')
            ax.axhline(0,linestyle='dotted',color='k')
            ax.axhline(1,linestyle='dotted',color='k')
            ax.legend(loc='best')
            st.pyplot(fig)            
        
    def biplot(self,loading_df,score_df,loading_color,score_color,axlim_pad=.55,n_loading_arrows=4,offset_scalar=1.3,figsz=(12,8)):
        '''
        Plots loading vectors vs. scores (xi projected onto loading vectors)
        Loading vectors reveal directions with largest variance in X
        Score vectors reveal (features xi projected onto loading vectors) reveal features xi in correspondence with 
        vectors with highest variance
        loading_df       : eigenvector table, DataFrame
        score_df         : features X projected onto loading dfs, DataFrame
        loading_color    : color of loading indices, str
        score_color      : color of scores indices, str
        axlim_pad        : expand the max_pc_val by this amount to display all sore annotations, float   
        n_loading_arrows : number of loading vector arrows to draw, int
        offset_scalar    : loading coordinate scalar multiple to offset feature annotations, float 
        returns          : pyplot figure
        '''
        with plt.style.context('seaborn-pastel'):
            fig = plt.figure(figsize=figsz)
            ax0 = fig.add_subplot()
            
            # step 1: plot teams score coordinates (x=PC1, y=PC2) values onto ax0  
            for team in score_df.index:
                ax0.annotate(team,(score_df['PC1'][team],-score_df['PC2'][team]),ha='center',color=score_color,label='Scores')
            # get max value of PC1,PC2 to scale ax0 xlim, ylim     
            max_pc_val = max(score_df['PC1'].abs().max(),score_df['PC2'].abs().max() ) + axlim_pad # add a little more to pad axis
            ax0.set(xlim=(-max_pc_val,max_pc_val),ylim=(-max_pc_val,max_pc_val))
#             ax0.set_title('Principal Components',fontsize=12,color=score_color)
            ax0.set_xlabel('1st Principal Component (team score values)',fontsize=12,color=score_color)
            ax0.set_ylabel('2nd Principal Component (team score values)',fontsize=12,color=score_color)
#             ax0.set_title('Loading Vectors',fontsize=12,color=loading_color)
            ax0.legend()
            # step 2: plot the ith loading vector value coordinates (L1[i],L2[i]) as directions of largest variance btwn L1,L2
            ax1= ax0.twinx().twiny()
            # note: place the feature annotation offset from the arrow head
            for feature in loading_df.index:
                ax1.annotate(feature,(loading_df['L1'][feature]*offset_scalar,-loading_df['L2'][feature]*offset_scalar),
                             color=loading_color,label='loadings')
                
            # get max value of PL1,L2 to scale ax1 xlim, ylim     
            max_loading_val = max( loading_df['L1'].abs().max(), loading_df['L2'].abs().max() ) +axlim_pad # add a little more to pad axis
            ax1.set( xlim=(-max_loading_val,max_loading_val), ylim=(-max_loading_val,max_loading_val) )
            # draw ith loading arrow from origin to coordinate (L1,L2)
            for i in range(n_loading_arrows):
                # matplotlib.pyplot.arrow(x, y, dx, dy, hold=None, **kwargs)  # e.g. kwargs 0.015,shape='full')
                ax1.arrow(x=0,y=0,dx=loading_df['L1'][i],dy=-loading_df['L2'][i], 
                          head_width=.015, shape='full')
#             ax1.legend()    
                    
        st.pyplot(fig)        
        
d = Data()
urldict = {
           'batting':'https://www.baseball-reference.com/leagues/MLB/{}-standard-batting.shtml',
           'pitching':'https://www.baseball-reference.com/leagues/MLB/{}-standard-pitching.shtml',
           'fielding':'https://www.baseball-reference.com/leagues/MLB/{}-standard-fielding.shtml',
           'standings_AL':'https://www.baseball-reference.com/leagues/AL/{}-standings.shtml',
           'standings_NL':'https://www.baseball-reference.com/leagues/NL/{}-standings.shtml',
}

d.addUrls(urldict)

thisYear = datetime.now().year

d.mlb_year = st.slider('Select a year:',
                1900,thisYear,thisYear)

st.title(f'MLB {d.mlb_year} Season')
# st.write(d.urldict['team_stats'])
d.BaseBallReferenceStandings(d.mlb_year)
# st.dataframe(d.standings_mlb_df)
d.BaseballReferenceTeamStatsYear(d.mlb_year)
d.division_standings() 
# st.dataframe(d.standings_league_div_dict['AL_E'])

s = Stats(d)
# scaled_df = s.center_scale(d.season_result_df)
eigvals,loading_df,score_df = s.PCA(d.season_result_df,'AL')
# st.dataframe(scaled_df)
s.PVE(eigvals,figsz=(12,6))
s.biplot(loading_df,score_df,score_color='b',loading_color='r',
         axlim_pad=.55,n_loading_arrows=len(loading_df),offset_scalar=1.275,figsz=(12,12))