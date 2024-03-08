import itertools
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (make_scorer, mean_absolute_percentage_error as MAPE)
from sklearn.model_selection import train_test_split

from data_synthesis import create_synthetic_data
from runtime_prediction import ErnestModel, GradientBoosting

plt.style.use('classic')
plt.rcParams['font.family'] = 'serif'

job_names = ('Sort', 'Grep', 'Linear Regression', 'K-Means', 'Page Rank')

def get_seed():
    get_seed.seed += 1
    return get_seed.seed
get_seed.seed = 0


def get_train_test_split(job_name, **kwargs):

    job_id = re.sub(r'[^a-zA-Z0-9]', '', job_name.lower())
    df = pd.read_csv(f'c3o_performance_data/{job_id}.csv')
    if job_id in ('kmeans', 'linearregression'):
        # The evaluated models are not equipped to model performance of configurations with large memory bottlenecks. Those configurations need and should not be offered to the user regardless
        df = df[~(df['instance_count'] <= 2)]
    X = df.drop('gross_runtime', axis=1)
    y = df['gross_runtime']
    return train_test_split(X, y, random_state=get_seed(), **kwargs)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))
MAPE = mean_absolute_percentage_error


def eval_experiment1(models, iterations=10):

    results = []

    for job_name, iteration in itertools.product(job_names, range(1, iterations+1)):

        print(f'Evaluating experiment1 on job "{job_name}", iteration {iteration}/{iterations}')

        X_train, X_test, y_train, y_test = get_train_test_split(job_name, test_size=.2)

        for model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            error = MAPE(y_test, y_pred)
            results.append((type(model).__name__, job_name, 'Original', 0, error))

        for num_synthetic_samples in range(100, 1101, 100):

            df_train = pd.concat([X_train, y_train], axis=1)
            df_synthetic = create_synthetic_data(df_train, num_synthetic_samples)
            X_train_syn = df_synthetic.drop('gross_runtime', axis=1)
            y_train_syn = df_synthetic['gross_runtime']

            for model in models:
                model.fit(X_train_syn, y_train_syn)
                y_pred = model.predict(X_test)
                error = MAPE(y_test, y_pred)
                results.append( (type(model).__name__, job_name, 'Synthetic',
                                 num_synthetic_samples, error) )

    columns = ['Model', 'Job', 'Mode', 'NumSyntheticSamples', 'MAPE']
    results = pd.DataFrame(results, columns=columns)
    results.to_csv('evaluation_results/experiment1.csv', index=False)


def eval_experiment2(models, iterations=10):

    results = []

    for job_name, iteration in itertools.product(job_names, range(1, iterations+1)):

        print(f'Evaluating experiment2 on job "{job_name}", iteration {iteration}/{iterations}')

        for num_original_samples in range(3, 34, 3):

            X_train, X_test, y_train, y_test = \
                    get_train_test_split(job_name, train_size=num_original_samples)

            for model in models:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                error = MAPE(y_test, y_pred)
                results.append( (type(model).__name__, job_name, 'Original',
                                 num_original_samples, 0, error) )

            df_train = pd.concat([X_train, y_train], axis=1)
            df_synthetic = create_synthetic_data(df_train, 1000)
            X_train_syn = df_synthetic.drop('gross_runtime', axis=1)
            y_train_syn = df_synthetic['gross_runtime']

            for model in models:
                model.fit(X_train_syn, y_train_syn)
                y_pred = model.predict(X_test)
                error = MAPE(y_test, y_pred)
                results.append( (type(model).__name__, job_name, 'Synthetic',
                                 num_original_samples, 1000, error) )

    columns = ['Model', 'Job', 'Mode', 'NumOriginalSamples', 'NumSyntheticSamples', 'MAPE']
    results = pd.DataFrame(results, columns=columns)
    results.to_csv('evaluation_results/experiment2.csv', index=False)


def eval_experiment3():

    results = []
    for num_original_samples, num_synthetic_samples, job_name in \
            itertools.product(range(10,31, 5), range(100,501,100), job_names):

        print(f'Evaluating experiment3 on job "{job_name}", {num_original_samples} original samples, '
              f' {num_synthetic_samples} synthetic samples.')

        start_time = time.time()
        X_train, _, y_train, __ = get_train_test_split(job_name, train_size=num_original_samples)
        df_train = pd.concat([X_train, y_train], axis=1)
        df_synthetic = create_synthetic_data(df_train, num_synthetic_samples)
        execution_duration = time.time() - start_time
        results.append( (num_original_samples, num_synthetic_samples, job_name,
                         len(df_synthetic.keys()), execution_duration) )

    columns = ['NumOriginalSamples', 'NumSyntheticSamples', 'Job', 'NumDatasetColumns', 'Runtime']
    results = pd.DataFrame(results, columns=columns)
    results.to_csv('evaluation_results/experiment3.csv', index=False)

def plot_experiment1_results(df):

    original_data = df['Mode'] == 'Original'
    synthetic_data = df['Mode'] == 'Synthetic'
    ernest_model = df['Model'] == 'ErnestModel'
    gradient_boosting = df['Model'] == 'GradientBoosting'

    for job_name in job_names:

        fig = plt.figure(figsize=(5,2.5))

        current_job = df['Job'] == job_name

        df_ori = df[current_job & ernest_model & original_data]

        df_syn = df[current_job & ernest_model & synthetic_data][['NumSyntheticSamples', 'MAPE']]\
                .groupby(['NumSyntheticSamples']).mean().reset_index()


        plt.plot((50, 1150), (df_ori['MAPE'].mean(), df_ori['MAPE'].mean()), color='tab:blue', linewidth=1.5, zorder=1)
        plt.scatter(df_syn['NumSyntheticSamples'], df_syn['MAPE'], color='tab:blue', s=100, marker=(4,1,0), zorder=2)

        df_ori = df[current_job & gradient_boosting & original_data]

        df_syn = df[current_job & gradient_boosting & synthetic_data][['NumSyntheticSamples', 'MAPE']]\
                .groupby(['NumSyntheticSamples']).mean().reset_index()

        plt.plot((50, 1150), (df_ori['MAPE'].mean(), df_ori['MAPE'].mean()), color='tab:green', linewidth=1.5, zorder=1)
        plt.scatter(df_syn['NumSyntheticSamples'], df_syn['MAPE'], color='tab:green', s=100, marker=(4,1,0), zorder=2)

        plt.title(job_name, fontsize=15, y=0.85, fontweight='bold', color='dimgray')
        plt.ylim(-0.02, 0.49); plt.xlim(50, 1150)
        plt.xticks(fontsize=12); plt.yticks(fontsize=12);
        plt.xlabel('Synthetic samples', fontsize=15, labelpad=-0.5);
        plt.ylabel('MAPE', fontsize=15, labelpad=-0.5);

        # Remove the top and right spines
        ax = plt.gca() # Get the current axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.tight_layout()
        plot_file_name = re.sub(r'[^a-zA-Z0-9]', '', job_name.lower())
        plt.savefig(f'evaluation_results/plots/experiment1_{plot_file_name}.pdf', bbox_inches='tight', pad_inches=0)


def plot_experiment2_results(df):

    original_data = df['Mode'] == 'Original'
    synthetic_data = df['Mode'] == 'Synthetic'
    ernest_model = df['Model'] == 'ErnestModel'
    gradient_boosting = df['Model'] == 'GradientBoosting'

    for job_name in job_names:

        fig = plt.figure(figsize=(5,2.5))

        current_job = df['Job'] == job_name

        df_ori = df[current_job & ernest_model & original_data][['NumOriginalSamples', 'MAPE']]\
                .groupby(['NumOriginalSamples']).mean().reset_index()
        df_syn = df[current_job & ernest_model & synthetic_data][['NumOriginalSamples', 'MAPE']]\
                .groupby(['NumOriginalSamples']).mean().reset_index()

        plt.plot(df_ori['NumOriginalSamples'], df_ori['MAPE'], color='tab:blue', linewidth=1.5, zorder=1)
        plt.scatter(df_syn['NumOriginalSamples'], df_syn['MAPE'], color='tab:blue', s=100, marker=(4,1,0), zorder=2)

        df_ori = df[current_job & gradient_boosting & original_data][['NumOriginalSamples', 'MAPE']]\
                .groupby(['NumOriginalSamples']).mean().reset_index()
        df_syn = df[current_job & gradient_boosting & synthetic_data][['NumOriginalSamples', 'MAPE']]\
                .groupby(['NumOriginalSamples']).mean().reset_index()

        plt.plot(df_ori['NumOriginalSamples'], df_ori['MAPE'], color='tab:green', linewidth=1.5, zorder=1)
        plt.scatter(df_syn['NumOriginalSamples'], df_syn['MAPE'], color='tab:green', s=100, marker=(4,1,0), zorder=2)

        plt.title(job_name, fontsize=15, y=0.85, fontweight='bold', color='dimgray')#, backgroundcolor='white')
        plt.ylim(-0.02, 0.69); plt.xlim(2, 34)
        plt.xticks(fontsize=12); plt.yticks(fontsize=12);
        plt.xlabel('Available original samples', fontsize=15, labelpad=-0.5);
        plt.ylabel('MAPE', fontsize=15, labelpad=-0.5);

        # Remove the top and right spines
        ax = plt.gca() # Get the current axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.tight_layout()
        plot_file_name = re.sub(r'[^a-zA-Z0-9]', '', job_name.lower())
        plt.savefig(f'evaluation_results/plots/experiment2_{plot_file_name}.pdf', bbox_inches='tight', pad_inches=0)


def plot_experiment3_results(df):

    def make_plot(xlabel):

        plt.ylim(-0.5, 13);
        xlim = (85, 515) if 'synthetic' in xlabel else (9.2, 30.8)
        plt.xlim(xlim)
        plt.xticks(fontsize=12); plt.yticks((0, 4, 8, 12), fontsize=12);
        plt.xlabel(xlabel, fontsize=14, labelpad=0.5);
        plt.ylabel('Runtime [s]', fontsize=14, labelpad=-0.5);

        # Remove the top and right spines
        ax = plt.gca() # Get the current axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')


    colors = {'Sort': 'tab:orange', 'Grep': 'tab:red', 'Linear Regression': 'tab:green', 'K-Means': 'pink', 'Page Rank':'tab:brown'}

    fig = plt.figure(figsize=(3.8,1.9))
    for i, job_name in enumerate(job_names):
        current_job = df['Job'] == job_name

        filtered_df = df[current_job][['NumSyntheticSamples', 'Runtime']]\
            .groupby(['NumSyntheticSamples']).mean().reset_index()

        plt.plot(filtered_df['NumSyntheticSamples'], filtered_df['Runtime'], linestyle=(i*2, (2.5,5)), color=colors[job_name], linewidth=5.5)

    make_plot('Generated synthetic samples')
    plt.tight_layout()
    plt.savefig(f'evaluation_results/plots/experiment3_1.pdf', bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(3.8,1.9))
    counter = 0
    for i, job_name in enumerate(job_names):
        current_job = df['Job'] == job_name

        filtered_df = df[current_job][['NumOriginalSamples', 'Runtime']]\
            .groupby(['NumOriginalSamples']).mean().reset_index()

        plt.plot(filtered_df['NumOriginalSamples'], filtered_df['Runtime'], linestyle=(i*2, (2.5,5)), color=colors[job_name], linewidth=5.5)

    make_plot('Available original samples')
    plt.tight_layout()
    plt.savefig(f'evaluation_results/plots/experiment3_2.pdf', bbox_inches='tight', pad_inches=0)

# Full eval with 10 iterations takes about 2h on a laptop
if not os.path.exists('evaluation_results/experiment1.csv'):
    eval_experiment1( (ErnestModel(), GradientBoosting()) )
df_experiment1 = pd.read_csv('evaluation_results/experiment1.csv')

if not os.path.exists('evaluation_results/experiment2.csv'):
    eval_experiment2( (ErnestModel(), GradientBoosting()) )
df_experiment2 = pd.read_csv('evaluation_results/experiment2.csv')

if not os.path.exists('evaluation_results/experiment3.csv'):
    eval_experiment3()
df_experiment3 = pd.read_csv('evaluation_results/experiment3.csv')


plot_experiment1_results(df_experiment1)
plot_experiment2_results(df_experiment2)
plot_experiment3_results(df_experiment3)

original = df_experiment2['Mode'] == 'Original'
synthetic = df_experiment2['Mode'] == 'Synthetic'
le30 = df_experiment2['NumOriginalSamples'] <= 30

error_ori = df_experiment2[original & le30]['MAPE'].mean()
error_syn = df_experiment2[synthetic & le30]['MAPE'].mean()
print(f'-- For less than 30 original samples --\n'
      f'Original data error: {error_ori}\n'
      f'Synthetic data error: {error_syn}\n'
      f'Difference: {abs((error_ori - error_syn) / error_ori)}'
      )

