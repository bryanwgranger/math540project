import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_dataset():
    df = pd.read_csv('US_counties_COVID19_health_weather_data.csv', parse_dates=['date'])
    df['cases_per_1000'] = df['cases'] / df['population'] * 1000
    df['deaths_per_1000'] = df['deaths'] / df['population'] * 1000
    return df

def filter_state(df, state):
    df_new = df[df['state'] == state]
    return df_new

def get_last_day_county(df):
    return df.sort_values(by='date').drop_duplicates(subset='county', keep='last')

def drop_all_na(df):
    return df.dropna(axis=1, how='all')

def corr_matrix(df, metric):
    return df.corr()[metric].sort_values(ascending=False)

def create_state_dataset(state):
    df = load_dataset()
    df_state = filter_state(df, state)
    return drop_all_na(get_last_day_county(df_state))

def create_national_dataset():
    df = load_dataset()
    
    big_df = pd.DataFrame()
    for state in df['state'].unique():
        df_state = get_last_day_county(filter_state(df, state))
        
        big_df = pd.concat([big_df, df_state])

    return drop_all_na(big_df)

def ridge_plots(alphas, X_train, y_train, X_test, y_test):
    r_scores = np.empty(len(alphas))
    r_mse = np.empty(len(alphas))
    for i, a in enumerate(alphas):
        ridge = Ridge(alpha=a)
        ridge.fit(X_train, y_train)
        r_scores[i] = ridge.score(X_test, y_test)
        r_mse[i] = mean_squared_error(ridge.predict(X_test), y_test)
        
    ridgecv = RidgeCV(alphas=alphas)
    ridgecv.fit(X_train, y_train)
    ridgecv_score = ridgecv.score(X_test, y_test)
    ridgecv_mse = mean_squared_error(ridgecv.predict(X_test), y_test)
    ridgecv_alpha = ridgecv.alpha_
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    fig.suptitle('R^2 Scores and MSE for Ridge Regressors')
    
    ax1.plot(alphas, r_scores, '-ko')
    ax1.axhline(ridgecv_score, color='b', ls='--')
    ax1.axvline(ridgecv_alpha, color='b', ls='--')
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel('Score')
    
    ax2.plot(alphas, r_mse, '-ko')
    ax2.axhline(ridgecv_mse, color='r', ls='--')
    ax2.axvline(ridgecv_alpha, color='r', ls='--')
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel('MSE')
    
    plt.subplots_adjust(wspace=.5)
    plt.savefig('ridge.png', dpi=150)
    plt.show()
    
def lasso_plots(alphas, X_train, y_train, X_test, y_test):
    l_scores = np.empty(len(alphas))
    l_mse = np.empty(len(alphas))
    for i, a in enumerate(alphas):
        lasso = Lasso(alpha=a)
        lasso.fit(X_train, y_train)
        l_scores[i] = lasso.score(X_test, y_test)
        l_mse[i] = mean_squared_error(lasso.predict(X_test), y_test)
        
    lassocv = LassoCV(alphas=alphas)
    lassocv.fit(X_train, y_train)
    lassocv_score = lassocv.score(X_test, y_test)
    lassocv_mse = mean_squared_error(lassocv.predict(X_test), y_test)
    lassocv_alpha = lassocv.alpha_
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    fig.suptitle('R^2 Scores and MSE for Lasso Optimizer')
    
    ax1.plot(alphas, l_scores, '-ko')
    ax1.axhline(lassocv_score, color='b', ls='--')
    ax1.axvline(lassocv_alpha, color='b', ls='--')
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel('Score')
    ax1.set_xscale('log')
    
    ax2.plot(alphas, l_mse, '-ko')
    ax2.axhline(lassocv_mse, color='r', ls='--')
    ax2.axvline(lassocv_alpha, color='r', ls='--')
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel('MSE')
    ax2.set_xscale('log')
    plt.subplots_adjust(wspace=.5)
    
    plt.savefig('lassofig.png', dpi=150)
    plt.show()
    
def elasticnet_plots(alphas, ratios, X_train, y_train, X_test, y_test):
    e_scores = np.empty(len(alphas))
    e_mse = np.empty(len(alphas))
    colors = ['b','g','r', 'c', 'm']
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    for ratio, color in zip(ratios, colors):    
        for i, a in enumerate(alphas):
            elas = ElasticNet(alpha=a, l1_ratio=ratio)
            elas.fit(X_train, y_train)
            e_scores[i] = elas.score(X_test, y_test)
            e_mse[i] = mean_squared_error(elas.predict(X_test), y_test)
        ax1.plot(alphas, e_scores, color=color, label=f'{ratio}', marker='o')
        ax2.plot(alphas, e_mse, color=color, label=f'{ratio}', marker='o')


    elnetcv = ElasticNetCV(alphas=alphas, l1_ratio=ratios)
    elnetcv.fit(X_train, y_train)
    ecv_score = elnetcv.score(X_test, y_test)
    ecv_l1_ratio = elnetcv.l1_ratio_
    ecv_alpha = elnetcv.alpha_
    ecv_mse = mean_squared_error(elnetcv.predict(X_test), y_test)
    
    
    
    ax1.axhline(ecv_score, color='k', ls='--')
    ax1.axvline(ecv_alpha, color='k', ls='--')
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel('Score')
    ax1.set_xscale('log')
    
    ax2.axhline(ecv_mse, color='k', ls='--')
    ax2.axvline(ecv_alpha, color='k', ls='--')
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel('MSE')
    ax2.set_xscale('log')
    fig.suptitle('R^2 Scores and MSE for Elastic Net Optimizer for each L1 Ratio')
    
    plt.subplots_adjust(wspace=.5)
    plt.legend()
    plt.savefig('elasticnet.png', dpi=150)
    plt.show()