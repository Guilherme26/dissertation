import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_word_relations(word, data):
    before = {}
    after = {}
    for sub in data:
        words = sub.split(' ')
        for i in range(len(words)):
            if words[i] == word:
                if i>0:
                    if words[i-1] not in before:
                        before[words[i-1]] = 1
                    else:
                        before[words[i-1]] += 1
                if i<len(words)-1:
                    if words[i+1] not in after:
                        after[words[i+1]] = 1
                    else:
                        after[words[i+1]] += 1

    before = pd.DataFrame().from_dict({'word':before.keys(),'count':before.values()})
    after = pd.DataFrame().from_dict({'word':after.keys(),'count':after.values()})
    
    return before, after

def plot_liwc_scored(sentences_liwc_counts_scored,ploting_vars,scores,plot_figsize=(16,2.8),max_one_line_plots=5,save_fig=None,comparative_behaviour=None):
    """
    Given the ploting vars, for each var this function plots the 
    distribution of the toxicity scores by sentence with and, without 
    the var or a random sample. This behaviour can be controlled by the
    comparative_behaviour argument.
    Args:
        sentences_liwc_counts_scored: dataframe with the LIWC features
        ploting_vars: list of features to plot
        scores: list of scores to plot
        plot_figsize: size of the figure
        max_one_line_plots: maximum number of features to plot in one line
        save_fig: str or None; if string (not empty), saves the figure with the given name
        comparative_behaviour: str, float or None; if str, it can be 'same_sz_sample' 
            (to use a random sample of same size) or 'same_sz_sample_with_var' (to use a random 
            sample of same size taken from data which the var is present). If float, 
            it is the size of the sample to use. If None, no comparison is made.
    """

    if len(ploting_vars)==0:
        raise Exception('No variables to plot')

    plot_dim = (
        -1 * ( - len(ploting_vars) // max_one_line_plots), 
        min(max_one_line_plots,len(ploting_vars))
    )
    # print(plot_dim)

    lengend_plot_i = -1 * ( - plot_dim[1] // 2) - 1

    for score in [i for i in scores if 'score' in i]:
        fig, axes = plt.subplots(plot_dim[0], plot_dim[1], sharey=True, sharex=False, figsize=plot_figsize)
        axes = axes.ravel()

        for i,var in enumerate(ploting_vars):
            series_present = sentences_liwc_counts_scored[sentences_liwc_counts_scored[var]>0][score]
            present_len = len(series_present)

            series_not_present = None
            if comparative_behaviour == None:
                series_not_present = sentences_liwc_counts_scored[sentences_liwc_counts_scored[var]==0][score]
            elif type(comparative_behaviour) == float and comparative_behaviour >= 0 and comparative_behaviour <= 1:
                series_not_present = sentences_liwc_counts_scored.sample(frac=comparative_behaviour)[score]
            elif type(comparative_behaviour) == str and comparative_behaviour == 'same_sz_sample':
                series_not_present = sentences_liwc_counts_scored.sample(n=present_len)[score]
            elif type(comparative_behaviour) == str and comparative_behaviour == 'same_sz_sample_not_present':
                series_not_present = sentences_liwc_counts_scored[sentences_liwc_counts_scored[var]==0].sample(n=present_len)[score]
            else:
                raise Exception('Comparative behaviour not recognized')
            not_present_len = len(series_not_present)

            if present_len == 0 or not_present_len == 0 :
                raise Exception('No sentences with non-zero or zero values for %s'%var)
                
            if i==lengend_plot_i:
                g2 = sns.ecdfplot(series_present,     complementary=True, ax=axes[i], legend=True, color='red')
                g1 = sns.ecdfplot(series_not_present, complementary=True, ax=axes[i], legend=True, color='black')

                new_labels = ['With','Without']
                g2.legend(new_labels)
                g1.legend_.set_title(None)
                for t, l in zip(g2.legend_.texts, new_labels):
                    t.set_text(l)
                g2.legend_.set_frame_on(False)
                g2.legend_.set_frame_on(False)
                
            else:
                sns.ecdfplot(series_present,          complementary=True, ax=axes[i], legend=False, color='red')
                g1 = sns.ecdfplot(series_not_present, complementary=True, ax=axes[i], legend=False, color='black')

            axes[i].set(ylabel=score+' - P[X>x]',xlabel='%s score - x'%var)

        fig.tight_layout()
        if save_fig:
            plt.savefig(filepaths['imgs']+score+save_fig)
        plt.show()

def plot_liwc_features(
    df, ploting_vars, 
    max_one_line_plots=5, 
    log_scale=False, 
    plot_figsize=(16,2.8), 
    save_fig=None,
    hue_order=['Black Woman','Black Man','White Woman', 'White Man'],
    legend_labels = ['BW', 'BM','WW','WM'],
    palette=['red','orange','grey','black']
):
    """
    Plots the LIWC features in a grid of plots.
    Args:
        df: dataframe with the LIWC features
        ploting_vars: list of features to plot
        max_one_line_plots: maximum number of features to plot in one line
        log_scale: if True, plots the features in log scale
        plot_figsize: size of the figure
        save_fig: str or None; if string (not empty), saves the figure with the given name
    """
    if len(ploting_vars)==0:
        raise Exception('No variables to plot')

    plot_dim = (
        -1 * ( - len(ploting_vars) // max_one_line_plots), 
        min(max_one_line_plots,len(ploting_vars))
    )
    # print(plot_dim)

    lengend_plot_i = -1 * ( - plot_dim[1] // 2) - 1

    fig, axes = plt.subplots(plot_dim[0], plot_dim[1], sharey=True, sharex=False, figsize=plot_figsize)
    axes = axes.ravel()

    for i,var in enumerate(ploting_vars): 
        if i==lengend_plot_i:
            g = sns.ecdfplot(
                df, 
                x=var, 
                hue="group",
                complementary=True, 
                ax=axes[i],
                legend=True,
                hue_order=hue_order,
                palette=palette
            ) 
            g.legend_.set_title(None)
            for t, l in zip(g.legend_.texts, legend_labels):
                t.set_text(l)
            g.legend_.set_frame_on(False)
            g.legend_.set_frame_on(False)
        else:
            sns.ecdfplot(
                df, 
                x=var, 
                hue="group",
                complementary=True, 
                ax=axes[i],
                legend=False, 
                hue_order=hue_order,
                palette=palette
            )

        axes[i].set(ylabel='P[X>x]',xlabel='%s score - x'%var)
        if log_scale:
            axes[i].set(yscale='log')

    fig.tight_layout()
    if save_fig:
        plt.savefig(filepaths['imgs']+save_fig)
        
    plt.show()