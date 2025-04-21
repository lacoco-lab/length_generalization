import numpy as np
from data_local import ape, nope
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

from data_local import random_baselines

def plot_with_legend():
    # Set default font to be bold
    # mpl.rcParams['font.weight'] = 'bold'
    # mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titleweight'] = 'bold'

    sns.set_theme(style="whitegrid", palette="dark6", context="paper", font_scale=2)

    # Example data
    x_labels = ['Bin 1', 'Bin 2', 'Bin 3']
    x = [10, 20, 30]

    data_ape = {
        'Algorithm': {'marker': '>', 'marker_size': 10, 'accuracies': [], 'names': [], 'expressive': []},
        'Star Free': {'marker': '>', 'marker_size': 10, 'accuracies': [], 'names': [], 'expressive': []},
        'Non Star Free': {'marker': '>', 'marker_size': 10, 'accuracies': [], 'names': [], 'expressive': []},
    }
    data_nope = {
        'Algorithm': {'marker': 'x', 'marker_size': 15, 'accuracies': [], 'names': [], 'expressive': []},
        'Star Free': {'marker': 'x', 'marker_size': 15, 'accuracies': [], 'names': [], 'expressive': []},
        'Non Star Free': {'marker': 'x', 'marker_size': 15, 'accuracies': [], 'names': [], 'expressive': []},
    }
    green_colors = ['seagreen', 'seagreen'] #lightgreen']
    red_colors = ['crimson', 'crimson'] #'lightcoral']
    line_styles = ['solid', 'dotted']
    # print(data_ape)
    # print(data_nope)
    for data_spec, ape_or_nope in zip([data_ape, data_nope], [ape, nope]):
        # Accuracy values for different datapoints
        for lang, info in ape_or_nope.items():
            # extract information from dictionary
            category = info['Type']
            if category == 'Counter':
                continue            
            name = info['name']
            expressive = info['Expressive']
            accuracy = (info['Bin 0'], info['Bin 1'], info['Bin 2'])

            # append to the data dictionary
            data_spec[category]['accuracies'].append(accuracy)
            data_spec[category]['names'].append(name)
            data_spec[category]['expressive'].append(expressive)

    # Separate algorithmic languages into individual subplots, all others in one consolidated subplot
    num_algo = len(data_ape['Algorithm']['names'])  # Number of individual algorithmic entries
    non_algorithmic_languages = ['Non Star Free', 'Star Free']

    y_limits = (-10, 110)

    # Create a figure with a custom gridspec layout
    fig = plt.figure(figsize=(20, 8))  # Adjust the size as necessary

    # Create a gridspec with 1 row and 2 columns, with relative width ratios (60% and 40%)
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 3], figure=fig)  # 3:2 width ratio

    # First half (60%): Divide into a (2, 4) grid
    gs_left = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0], wspace=0.4, hspace=0.4)

    # Second half (40%): Divide into a (3, 6) grid
    gs_right = gridspec.GridSpecFromSubplotSpec(3, 6, subplot_spec=gs[1], wspace=0.3, hspace=0.3)

    gs_left_half = 4  # number of columns
    gs_right_third = 6  # number of columns
    # Plot each individual algorithm in a separate subplot
    sub_plot_dict = {}
    for over_idx, data_ape_or_nope in enumerate([data_ape, data_nope]):
        for idx, (name, accuracies, expressive) in enumerate(zip(data_ape_or_nope['Algorithm']['names'], data_ape_or_nope['Algorithm']['accuracies'], data_ape_or_nope['Algorithm']['expressive'])):
            
            i = idx // gs_left_half  # Row index (0 or 1)
            j = idx % gs_left_half  # Column index (0 to 3)
            if over_idx == 0:
                ax = fig.add_subplot(gs_left[i , j])
                sub_plot_dict[idx] = ax
            else: 
                ax = sub_plot_dict[idx]
            # print(idx, i, j)

            color = green_colors[over_idx] if expressive else red_colors[over_idx]
            
            # Line style - solid for APE ; dotted for NOP
            # Plot each algorithm separately
            ax.plot(x, accuracies, marker=data_ape_or_nope['Algorithm']['marker'], linestyle=line_styles[over_idx], color=color, markersize=data_ape['Algorithm']['marker_size'])
            ax.plot(x, [random_baselines[name] for _ in range(3)], marker='o', linestyle='dashed', color='gray')
            if over_idx != 0:
                ax.set_title(f'{name}')
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels)
                ax.set_ylim(y_limits)
                # ax.set_xlabel('Validations')
                # ax.set_ylabel('Accuracy')
                ax.grid(True)


    # Plot on the right side (3x6 subplots, with the last one for the legend)
    sub_plot_dict = {}
    for over_idx, data_ape_or_nope in enumerate([data_ape, data_nope]):
        count = 1
        for category in non_algorithmic_languages:
            for idx, (name, accuracies, expressive) in enumerate(zip(data_ape_or_nope[category]['names'], data_ape_or_nope[category]['accuracies'], data_ape_or_nope[category]['expressive'])):
                
                i = (count - 1)  // gs_right_third  # Row index (0, 1, or 2)
                j = (count - 1) % gs_right_third  # Column index (0 to 5)
                #print()
                if over_idx == 0:
                    ax = fig.add_subplot(gs_right[i , j])
                    sub_plot_dict[count] = ax
                else: 
                    ax = sub_plot_dict[count]
                # ax = fig.add_subplot(gs_right[i, j])
                color = green_colors[over_idx] if expressive else red_colors[over_idx]
            
                # Line style - solid for APE ; dotted for NOPE
                # Plot each algorithm separately
                ax.plot(x, accuracies, marker=data_ape_or_nope[category]['marker'], linestyle=line_styles[over_idx], color=color, markersize=data_ape[category]['marker_size'])

                if over_idx == 0:
                    # UNCOMMENT BELOW LINE TO ADD NAMES TO EACH OF THE FORMAL LANG GRAPHS
                    # ax.set_title(f'{name}', fontsize=12)
                    ax.tick_params(axis='both', which='major', labelsize=8)  # Reduce label size to 6
                    ax.set_xticks(x)
                    ax.set_xticklabels([])
                    ax.set_ylim(y_limits)
                    ax.set_yticklabels([])
                    ax.grid(True)
                    ax.text(0.5, 0.5, f'{count}', ha='center', va='center', transform=ax.transAxes)
                count += 1

    # Add a legend for the different line styles
    legend_entries = [
        mlines.Line2D([], [], color='seagreen', linestyle='solid', label='Found CRASP[Periodic, Local] Program', marker='>'),
        mlines.Line2D([], [], color='crimson', linestyle='solid', label='No CRASP[Periodic, Local] Program', marker='X'),
        mlines.Line2D([], [], color='seagreen', linestyle='dotted', label='Found CRASP[] Program', marker='>'),
        mlines.Line2D([], [], color='crimson', linestyle='dotted', label='No CRASP[] Program', marker='X'),
    ]
    # Create a custom legend at the top of the figure, outside the subplots
    # increase font size
    fig.legend(handles=legend_entries, loc='upper center', ncol=4, frameon=False, fontsize=18)

    # Adjust the layout to ensure no overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reduce the bottom rect to fit the legend at the top
    # print("what happened")
    plt.savefig('results_for_main_paper.pdf', bbox_inches="tight")
    # plt.show()

# Call the function to plot the figure
plot_with_legend()
