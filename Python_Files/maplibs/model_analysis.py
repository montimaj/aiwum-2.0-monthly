# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    pred_df = pd.read_csv('../../Models/Predictions_BRF.csv').sort_values(by='Latitude', ascending=False)
    pred_df['Actual_AF'] *= 0.3048
    pred_df['Pred_AF'] *= 0.3048
    alpha_values = [0.9, 0.95, 0.8, 0.7, 0.6, 0.55, 0.3]
    train_df = pred_df[pred_df['DATA'] == 'TRAIN']
    test_df = pred_df[pred_df['DATA'] == 'TEST']
    plt.rcParams.update({'font.size': 18})
    #marker_scale = 13
    figsize = (12, 10)
    pal = sns.color_palette('Paired')
    markers = ['o', 'X', 's', 'P', 'D', 'p', '^']
    s = range(30, 16, -2)

    fig, ax = plt.subplots(figsize=figsize)
    for idx, year in enumerate(sorted(train_df['ReportYear'].unique())):
         sub_train_df = train_df[train_df['ReportYear'] == year]
         k = pal[idx]
         if idx > 0 and idx % 2 == 0:
             k = pal[idx + 1]
         sns.scatterplot(data=sub_train_df, y='Actual_AF', x='Pred_AF', ax=ax, marker=markers[idx], palette=k,
                         alpha=alpha_values[idx], label=year, s=s[idx])
    # sns.scatterplot(data=train_df, y='Actual_AF', x='Pred_AF', hue='ReportYear', ax=ax,
    #                  palette="Paired", style='ReportYear')
    sns.lineplot(data=train_df, x='Actual_AF', y='Actual_AF', color='k', ax=ax, label='1:1 relationship')
    ax.set_ylabel('Actual Water Use (m)')
    ax.set_xlabel('Predicted Water Use (m)')
    plt.title('Training Data', fontweight="bold")
    # plt.show()
    h, l = ax.get_legend_handles_labels()
    plt.legend(h, l, ncol=2)
    plt.show()
    # plt.savefig('../../Outputs/Train_Scatter.png', dpi=600)

    # fig, ax = plt.subplots(figsize=figsize)
    # sns.scatterplot(data=test_df, y='Actual_AF', x='Pred_AF', hue='ReportYear', ax=ax,
    #                 palette="Paired", style='ReportYear', alpha=0.7)
    # sns.lineplot(data=test_df, x='Actual_AF', y='Actual_AF', color='k', ax=ax, label='1:1 relationship')
    # ax.set_ylabel('Actual Water Use (m)')
    # ax.set_xlabel('Predicted Water Use (m)')
    # plt.title('Test Data', fontweight="bold")
    # # plt.show()
    # h, l = ax.get_legend_handles_labels()
    # plt.legend(h, l, ncol=2)
    # plt.savefig('../../Outputs/Test_Scatter.png', dpi=600)
