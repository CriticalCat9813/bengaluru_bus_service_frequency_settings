import pandas as pd
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import *


def plot_convergence(ns, objectives, timeLimit, epsLimit, iterLimit, showlegend=False):

    data = []
    for n in ns:
        for objective in objectives:
            modelname = '{0}-{1}-{2}-{3}'.format(n, objective, timeLimit, epsLimit)
            for iterCount in range(iterLimit + 1):
                try:
                    with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(
                            RELPATH, FILENAME, modelname, iterCount
                    ), 'rb') as file:
                        _, u_N, tt, cutct, eps, S, kappa = pickle.load(file)
                        tt /= (60 * 60)
                        util_obj = sum(u_N.values())
                        util_obj /= len(u_N)
                        mxmn_obj = min(u_N.values())
                        min_S_util = min(u_N[i] for i in S)
                        max_S_util = max(u_N[i] for i in S)
                        data.append(
                            (
                                n, objective, timeLimit, iterCount,
                                tt, cutct, kappa, eps, util_obj, mxmn_obj, min_S_util, max_S_util
                            )
                        )
                except FileNotFoundError:
                    pass

    df = pd.DataFrame(
        data,
        columns=[
            'n', 'objective', 'timelimit', 'iterCount',
            'tt', 'cutct', 'kappa', 'eps', 'util_obj', 'mxmn_obj', 'min_S_util', 'max_S_util']
    )

    color_map = {n: HEXCOLOR for n, HEXCOLOR in zip(ns, HEXCOLORS)}
    marker_map = {n: MARKER for n, MARKER in zip(ns, MARKERS)}
    subplot_map = {objective: idx + 1 for idx, objective in enumerate(objectives)}

    # plots = (
    #     ('eps', r'$\large \textrm{Multiplicative Least Objection (}\epsilon\textrm{)}$'),
    #     ('mxmn_obj', r'$\large \textrm{Maximin Social Welfare}$'),
    #     ('kappa', r'$\large \textrm{Basis Condition Number}$')
    # )

    plots = (
        ('tt', r'$\large \textrm{Elapsed time [hr.]}$'),
        ('cutct', r'$\large \textrm{Number of Cuts}$'),
        ('kappa', r'$\large \textrm{Basis Condition Number}$'),
        ('eps', r'$\large \textrm{Multiplicative Least Objection (}\epsilon\textrm{)}$'),
        ('util_obj', r'$\large \textrm{Utilitarian Social Welfare}$'),
        ('mxmn_obj', r'$\large \textrm{Maximin Social Welfare}$'),
        ('min_S_util', r'$\large \textrm{min_S_util}$'),
        ('max_S_util', r'$\large \textrm{max_S_util}$')
    )

    for col, title in plots:
        fig = make_subplots(
            rows=1, cols=len(objectives), shared_xaxes=True, shared_yaxes=True,
            subplot_titles=(r'$\Large \textrm{Maximin Service Goal}$', r'$\Large \textrm{Utilitarian Service Goal}$')
        )
        for (objective, n), group_df in df.groupby(['objective', 'n']):
            fig.add_trace(
                go.Scatter(
                    x=group_df['iterCount'],
                    y=group_df[col],
                    mode='lines+markers',
                    name=r'$n: {0}$'.format(n),
                    showlegend=True if subplot_map[objective] == 1 and showlegend else False,
                    line={'color': color_map[n], 'dash': 'solid'},
                    marker={'color': color_map[n], 'symbol': marker_map[n], 'size': 5}
                ),
                row=1,
                col=subplot_map[objective]
            )
        if col in {'kappa'}:
            fig.update_yaxes(type='log')
        fig.update_yaxes(
            row=1, col=1,
            title_text=title, title_font={'size': 18}
        )
        for idx, _ in enumerate(objectives):
            fig.update_xaxes(
                row=1, col=idx + 1,
                range=[0, iterLimit], title_text=r'$\large \textrm{Number of Iterations}$', title_font={'size': 18}
            )
        for annotation in fig['layout']['annotations']:
            annotation['font'] = {'size': 22}
            annotation['y'] = 1.0125
        fig.update_layout(
            legend={
                'orientation': 'h', 'entrywidth': 200, 'yanchor': 'top', 'y': -0.125, 'xanchor': 'left', 'x': 0,
                'font': {'size': 14}
            }
        )

        fig.show()
        fig.write_image('./results/figures/{0}.png'.format(col), width=1000, height=500, scale=4)


def plot_utilities(title, keys):

    data = []
    for n, objective, timeLimit, epsLimit, iterCount, subtitle in keys:
        modelname = '{0}-{1}-{2}-{3}'.format(n, objective, timeLimit, epsLimit)
        try:
            with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(
                    RELPATH, FILENAME, modelname, iterCount
            ), 'rb') as file:
                _, u_N, _, _, _, _, _ = pickle.load(file)
                u_N = sorted(u_N.values())
                for i, u_i in enumerate(u_N):
                    data.append(
                        (
                            n, objective, timeLimit, epsLimit, iterCount, subtitle, i + 1, u_i
                        )
                    )
        except FileNotFoundError:
            pass

    df = pd.DataFrame(
        data,
        columns=['n', 'objective', 'timeLimit', 'epsLimit', 'iterCount', 'title', 'i', 'u_i']
    )

    color_map = {key: HEXCOLOR for key, HEXCOLOR in zip(keys, HEXCOLORS)}
    marker_map = {key: MARKER for key, MARKER in zip(keys, MARKERS)}

    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=(title,)
    )

    for key, group_df in df.groupby(['n', 'objective', 'timeLimit', 'epsLimit', 'iterCount', 'title']):
        _, _, _, _, _, subtitle = key
        fig.add_trace(
            go.Scatter(
                x=group_df['i'],
                y=group_df['u_i'],
                mode='lines+markers',
                name=subtitle,
                line={'color': color_map[key], 'dash': 'solid'},
                marker={'color': color_map[key], 'symbol': marker_map[key], 'size': 5}
            ),
            row=1, col=1
        )

    fig.update_yaxes(
        row=1, col=1,
        title_text=r'$\large \textrm{Utility}$', title_font={'size': 18}
    )
    fig.update_xaxes(
        row=1, col=1,
        title_text=r'$\large \textrm{Riders (Sorted by Utility)}$', title_font={'size': 18}
    )
    for annotation in fig['layout']['annotations']:
        annotation['font'] = {'size': 22}
        annotation['y'] = 1.0125
    fig.update_layout(
        legend={
            'orientation': 'h', 'entrywidth': 200, 'yanchor': 'top', 'y': -0.125, 'xanchor': 'left', 'x': 0,
            'font': {'size': 14}
        }
    )

    fig.show()
    fig.write_image('./results/figures/utils.png', width=800, height=600, scale=4)


if __name__ == '__main__':

    ns = [1430]
    objectives = ['maximin', 'utilitarian']
    timeLimit = 300
    epsLimit = 1
    iterLimit = 101
    plot_convergence(ns, objectives, timeLimit, epsLimit, iterLimit)

    title = r'$\Large \textrm{Maximin Service Goal}$'
    keys = [
        (1430, 'maximin', 300, 1, 101, r'$\textrm{With cooperation}$'),
        (1430, 'maximin', 300, 1, -1, r'$\textrm{Without cooperation}$')
    ]
    plot_utilities(title, keys)

    title = r'$\Large \textrm{Utilitarian Service Goal}$'
    keys = [
        (1430, 'utilitarian', 300, 1, 101, r'$\textrm{With cooperation}$'),
        (1430, 'utilitarian', 300, 1, -1, r'$\textrm{Without cooperation}$')
    ]
    plot_utilities(title, keys)
