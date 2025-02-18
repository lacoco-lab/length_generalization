import numpy as np
from data_local import ape, nope, reg
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors

def plot_with_legend():
    # Set default font to be bold
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titleweight'] = 'bold'

    sns.set_theme(style="whitegrid", palette="dark6", context="paper", font_scale=2.3)

    # Example data
    x_labels = ['Bin 1', 'Bin 2', 'Bin 3']
    x = [10, 20, 30]

    # Data structure only for Star Free and Non Star Free categories (no Algorithm)
    data_ape = {
        'Star Free': {'marker': '>', 'marker_size': 10, 'accuracies': [], 'names': [], 'expressive': []},
        'Non Star Free': {'marker': '>', 'marker_size': 10, 'accuracies': [], 'names': [], 'expressive': []},
    }
    data_nope = {
        'Star Free': {'marker': 'x', 'marker_size': 15, 'accuracies': [], 'names': [], 'expressive': []},
        'Non Star Free': {'marker': 'x', 'marker_size': 15, 'accuracies': [], 'names': [], 'expressive': []},
    }
    data_reg = {
        'Star Free': {'marker': '+', 'marker_size': 15, 'accuracies': [], 'names': [], 'expressive': []},
        'Non Star Free': {'marker': '+', 'marker_size': 15, 'accuracies': [], 'names': [], 'expressive': []},
    }    

    green_colors = ['seagreen', 'seagreen', 'seagreen']
    red_colors = ['crimson', 'crimson', 'crimson']
    line_styles = ['solid', 'dotted', 'dashed']

    # Populate data for only Star Free and Non Star Free categories
    for data_spec, ape_or_nope in zip([data_ape, data_nope, data_reg], [ape, nope, reg]):
        for lang, info in ape_or_nope.items():
            category = info['Type']
            name = info['name']
            if category not in data_spec:  # Skip Counter as before
                continue            
            expressive = info['Expressive']
            accuracy = [info['Bin 0'], info['Bin 1'], info['Bin 2']]
            data_spec[category]['accuracies'].append(accuracy)
            data_spec[category]['names'].append(name)
            data_spec[category]['expressive'].append(expressive)

    non_algorithmic_languages = ['Non Star Free', 'Star Free']
    y_limits = (-10, 110)

    # Create a figure and a (3x6) grid of subplots using plt.subplots
    fig, axes = plt.subplots(3, 6, figsize=(20, 8), constrained_layout=True)
    axes = axes.ravel()  # Flatten the axes array to make indexing easier

    # Plot on the right side (3x6 subplots)
    for over_idx, data_ape_or_nope in enumerate([data_ape, data_nope, data_reg]):
        count = 0
        for category in non_algorithmic_languages:
            for idx, (name, accuracies, expressive) in enumerate(zip(data_ape_or_nope[category]['names'], data_ape_or_nope[category]['accuracies'], data_ape_or_nope[category]['expressive'])):
                ax = axes[count]  # Select the appropriate axis
                color = green_colors[over_idx] if expressive else red_colors[over_idx]
            
                # Line style - solid for APE; dotted for NOPE
                ax.plot(x, accuracies, marker=data_ape_or_nope[category]['marker'], linestyle=line_styles[over_idx], color=color, markersize=data_ape[category]['marker_size'])

                # Optionally add titles and adjust appearance
                ax.set_title(f'{name}')
                # ax.tick_params(axis='both', which='major', labelsize=8)  # Reduce label size
                ax.set_xticks(x)
                ax.set_xticklabels(['Bin 1', 'Bin 2', 'Bin 3'])
                ax.set_ylim(y_limits)
                # ax.set_yticklabels([])
                ax.grid(True)
                
                count += 1

    ax = axes[-1]
    ax.axis('off')  # Turn off the last subplot
    # Add a legend for the different line styles
    legend_entries = [
        mlines.Line2D([], [], color='seagreen', linestyle='solid', label='Found CRASP[Periodic, Local] Program', marker='>'),
        mlines.Line2D([], [], color='crimson', linestyle='solid', label='No CRASP[Periodic, Local] Program', marker='X'),
        mlines.Line2D([], [], color='seagreen', linestyle='dotted', label='Found CRASP[] Program', marker='>'),
        mlines.Line2D([], [], color='crimson', linestyle='dotted', label='No CRASP[] Program', marker='X'),
    ]
    # Create a custom legend at the top of the figure, outside the subplots
    fig.legend(handles=legend_entries, loc='upper center', ncol=4, frameon=False)

    # Save or display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reduce the bottom rect to fit the legend at the top
    # plt.tight_layout()
    # plt.show()
    plt.savefig('formal_only.pdf', bbox_inches="tight")


def dot_depth():
    # Set default font to be bold
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titleweight'] = 'bold'

    sns.set_theme(style="whitegrid", palette="dark6", context="paper", font_scale=2.4)

    # Example data
    x_labels = ['Bin 1', 'Bin 2', 'Bin 3']
    x = [10, 20, 30]

    # Data structure only for Star Free and Non Star Free categories (no Algorithm)
    data_ape = {
        'Star Free': {'marker': 'o', 'marker_size': 10, 'accuracies': [], 'names': [], 'expressive': []},
        'Non Star Free': {'marker': 'X', 'marker_size': 10, 'accuracies': [], 'names': [], 'expressive': []},
    }
    data_nope = {
        'Star Free': {'marker': 'o', 'marker_size': 15, 'accuracies': [], 'names': [], 'expressive': []},
        'Non Star Free': {'marker': 'X', 'marker_size': 15, 'accuracies': [], 'names': [], 'expressive': []},
    }

    green_colors = ['seagreen', 'seagreen']
    red_colors = ['crimson', 'crimson']
    line_styles = ['solid', 'dotted']

    # Populate data for only Star Free and Non Star Free categories
    for data_spec, ape_or_nope in zip([data_ape, data_nope], [ape, nope]):
        for lang, info in ape_or_nope.items():
            category = info['Type']
            name = info['name']
            if category not in data_spec:  # Skip Counter as before
                continue            
            accuracy = [info['Bin 0'], info['Bin 1'], info['Bin 2']]
            data_spec[category]['accuracies'].append(accuracy)
            data_spec[category]['names'].append(name)

    non_algorithmic_languages = ['Non Star Free', 'Star Free']
    y_limits = (-10, 110)

    # Create a figure and a (3x6) grid of subplots using plt.subplots
    fig, axes = plt.subplots(3, 6, figsize=(20, 8), constrained_layout=True)
    axes = axes.ravel()  # Flatten the axes array to make indexing easier

    # Plot on the right side (3x6 subplots)
    for over_idx, data_ape_or_nope in enumerate([data_ape, data_nope]):
        count = 0
        for category in non_algorithmic_languages:
            star_free = (category == 'Star Free')
            for idx, (name, accuracies) in enumerate(zip(data_ape_or_nope[category]['names'], data_ape_or_nope[category]['accuracies'])):
                ax = axes[count]  # Select the appropriate axis
                
                color = green_colors[over_idx] if star_free else red_colors[over_idx]
            
                # Line style - solid for APE; dotted for NOPE
                ax.plot(x, accuracies, marker=data_ape_or_nope[category]['marker'], linestyle=line_styles[over_idx], color=color, markersize=data_ape[category]['marker_size'])

                # Optionally add titles and adjust appearance
                ax.set_title(f'{name}')
                # ax.tick_params(axis='both', which='major', labelsize=8)  # Reduce label size
                ax.set_xticks(x)
                ax.set_xticklabels(['Bin 1', 'Bin 2', 'Bin 3'])
                ax.set_ylim(y_limits)
                # ax.set_yticklabels([])
                ax.grid(True)
                
                count += 1

    ax = axes[-1]
    ax.axis('off')  # Turn off the last subplot
    # Add a legend for the different line styles
    legend_entries = [
        mlines.Line2D([], [], color='seagreen', linestyle='solid', label='Star Free + APE', marker='o'),
        mlines.Line2D([], [], color='crimson', linestyle='solid', label='Non Star Free + APE', marker='X'),
        mlines.Line2D([], [], color='seagreen', linestyle='dotted', label='Star Free + NoPE', marker='o'),
        mlines.Line2D([], [], color='crimson', linestyle='dotted', label='Non Star Free + NoPE', marker='X'),
    ]
    # Create a custom legend at the top of the figure, outside the subplots
    fig.legend(handles=legend_entries, loc='upper center', ncol=4, frameon=False)

    # Save or display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reduce the bottom rect to fit the legend at the top
    # plt.tight_layout()
    # plt.show()
    plt.savefig('dot_depth_draft.pdf', bbox_inches="tight")


def ac0_vs_lengen():
    # Set default font to be bold
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titleweight'] = 'bold'

    sns.set_theme(style="whitegrid", palette="dark6", context="paper", font_scale=2.4)

    # Example data
    x_labels = ['Bin 1', 'Bin 2', 'Bin 3']
    x = [10, 20, 30]

    # Data structure only for Star Free and Non Star Free categories (no Algorithm)
    data_ape = {
        'Star Free': {'marker': '>', 'marker_size': 10, 'accuracies': [], 'names': [], 'expressive': []},
        'Non Star Free': {'marker': '>', 'marker_size': 10, 'accuracies': [], 'names': [], 'expressive': []},
    }
    data_nope = {
        'Star Free': {'marker': 'x', 'marker_size': 15, 'accuracies': [], 'names': [], 'expressive': []},
        'Non Star Free': {'marker': 'x', 'marker_size': 15, 'accuracies': [], 'names': [], 'expressive': []},
    }

    green_colors = ['seagreen', 'seagreen']
    red_colors = ['crimson', 'crimson']
    line_styles = ['solid', 'dotted']

    # Populate data for only Star Free and Non Star Free categories
    for data_spec, ape_or_nope in zip([data_ape, data_nope], [ape, nope]):
        for lang, info in ape_or_nope.items():
            category = info['Type']
            name = info['name']
            if category not in data_spec:  # Skip Counter as before
                continue            
            expressive = info['ac0']
            accuracy = [info['Bin 0'], info['Bin 1'], info['Bin 2']]
            data_spec[category]['accuracies'].append(accuracy)
            data_spec[category]['names'].append(name)
            data_spec[category]['expressive'].append(expressive)

    non_algorithmic_languages = ['Non Star Free', 'Star Free']
    y_limits = (-10, 110)

    # Create a figure and a (3x6) grid of subplots using plt.subplots
    fig, axes = plt.subplots(3, 6, figsize=(20, 8), constrained_layout=True)
    axes = axes.ravel()  # Flatten the axes array to make indexing easier

    # Plot on the right side (3x6 subplots)
    for over_idx, data_ape_or_nope in enumerate([data_ape, data_nope]):
        count = 0
        for category in non_algorithmic_languages:
            for idx, (name, accuracies, expressive) in enumerate(zip(data_ape_or_nope[category]['names'], data_ape_or_nope[category]['accuracies'], data_ape_or_nope[category]['expressive'])):
                ax = axes[count]  # Select the appropriate axis
                color = green_colors[over_idx] if expressive else red_colors[over_idx]
            
                # Line style - solid for APE; dotted for NOPE
                ax.plot(x, accuracies, marker=data_ape_or_nope[category]['marker'], linestyle=line_styles[over_idx], color=color, markersize=data_ape[category]['marker_size'])

                # Optionally add titles and adjust appearance
                ax.set_title(f'{name}')
                # ax.tick_params(axis='both', which='major', labelsize=8)  # Reduce label size
                ax.set_xticks(x)
                ax.set_xticklabels(['Bin 1', 'Bin 2', 'Bin 3'])
                ax.set_ylim(y_limits)
                # ax.set_yticklabels([])
                ax.grid(True)
                
                count += 1

    ax = axes[-1]
    ax.axis('off')  # Turn off the last subplot
    # Add a legend for the different line styles
    legend_entries = [
        mlines.Line2D([], [], color='seagreen', linestyle='solid', label='In AC0'),
        mlines.Line2D([], [], color='crimson', linestyle='solid', label='Not In AC0'),
        # mlines.Line2D([], [], color='seagreen', linestyle='dotted', label='In AC0', marker='>'),
        # mlines.Line2D([], [], color='crimson', linestyle='dotted', label='Not In AC0', marker='X'),
    ]
    # Create a custom legend at the top of the figure, outside the subplots
    fig.legend(handles=legend_entries, loc='upper center', ncol=4, frameon=False)

    # Save or display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reduce the bottom rect to fit the legend at the top
    # plt.tight_layout()
    # plt.show()
    plt.savefig('ac0.pdf', bbox_inches="tight")


def ac0_vs_lengen_algo():
    # Set default font to be bold
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titleweight'] = 'bold'

    sns.set_theme(style="whitegrid", palette="dark6", context="paper", font_scale=2)

    # Example data
    x_labels = ['Bin 1', 'Bin 2', 'Bin 3']
    x = [10, 20, 30]

    # Data structure only for Star Free and Non Star Free categories (no Algorithm)
    data_ape = {
        'Algorithm': {'marker': '>', 'marker_size': 10, 'accuracies': [], 'names': [], 'expressive': []},
        # 'Non Star Free': {'marker': '>', 'marker_size': 10, 'accuracies': [], 'names': [], 'expressive': []},
    }
    data_nope = {
        'Algorithm': {'marker': 'x', 'marker_size': 15, 'accuracies': [], 'names': [], 'expressive': []},
        # 'Non Star Free': {'marker': 'x', 'marker_size': 15, 'accuracies': [], 'names': [], 'expressive': []},
    }

    green_colors = ['seagreen', 'seagreen']
    red_colors = ['crimson', 'crimson']
    line_styles = ['solid', 'dotted']

    # Populate data for only Star Free and Non Star Free categories
    for data_spec, ape_or_nope in zip([data_ape, data_nope], [ape, nope]):
        for lang, info in ape_or_nope.items():
            category = info['Type']
            name = info['name']
            if category not in data_spec:  # Skip Counter as before
                continue            
            expressive = info['ac0']
            accuracy = [info['Bin 0'], info['Bin 1'], info['Bin 2']]
            data_spec[category]['accuracies'].append(accuracy)
            data_spec[category]['names'].append(name)
            data_spec[category]['expressive'].append(expressive)

    non_algorithmic_languages = ['Algorithm']
    y_limits = (-10, 110)

    # Create a figure and a (3x6) grid of subplots using plt.subplots
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    axes = axes.ravel()  # Flatten the axes array to make indexing easier

    # Plot on the right side (3x6 subplots)
    for over_idx, data_ape_or_nope in enumerate([data_ape, data_nope]):
        count = 0
        for category in non_algorithmic_languages:
            for idx, (name, accuracies, expressive) in enumerate(zip(data_ape_or_nope[category]['names'], data_ape_or_nope[category]['accuracies'], data_ape_or_nope[category]['expressive'])):
                ax = axes[count]  # Select the appropriate axis
                color = green_colors[over_idx] if expressive else red_colors[over_idx]
            
                # Line style - solid for APE; dotted for NOPE
                ax.plot(x, accuracies, marker=data_ape_or_nope[category]['marker'], linestyle=line_styles[over_idx], color=color, markersize=data_ape[category]['marker_size'])

                # Optionally add titles and adjust appearance
                ax.set_title(f'{name}')
                # ax.tick_params(axis='both', which='major', labelsize=8)  # Reduce label size
                ax.set_xticks(x)
                ax.set_xticklabels(['Bin 1', 'Bin 2', 'Bin 3'])
                ax.set_ylim(y_limits)
                # ax.set_yticklabels([])
                ax.grid(True)
                
                count += 1

    # ax = axes[-1]
    # ax.axis('off')  # Turn off the last subplot
    # Add a legend for the different line styles
    legend_entries = [
        mlines.Line2D([], [], color='seagreen', linestyle='solid', label='In AC0'),
        mlines.Line2D([], [], color='crimson', linestyle='solid', label='Not In AC0'),
        # mlines.Line2D([], [], color='seagreen', linestyle='dotted', label='In AC0 (NoPE Results)', marker='>'),
        # mlines.Line2D([], [], color='crimson', linestyle='dotted', label='Not In AC0 (NoPE Results)', marker='X'),
    ]
    # Create a custom legend at the top of the figure, outside the subplots
    fig.legend(handles=legend_entries, loc='upper center', ncol=4, frameon=False)

    # Save or display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reduce the bottom rect to fit the legend at the top
    # plt.tight_layout()
    # plt.show()
    plt.savefig('ac0_algo.pdf', bbox_inches="tight")


def dotdepth_vs_lengen():
    # Set default font to be bold
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titleweight'] = 'bold'

    sns.set_theme(style="whitegrid", palette="dark6", context="paper", font_scale=2.4)

    # Example data
    x_labels = ['Bin 1', 'Bin 2', 'Bin 3']
    line_styles = ['solid', 'dotted']
    x = [10, 20, 30]

    # Data structure only for Star Free and Non Star Free categories (no Algorithm)
    data_ape = {
        'Star Free': {'marker': '>', 'marker_size': 10, 'accuracies': [], 'names': [], 'expressive': []},
        'Non Star Free': {'marker': '>', 'marker_size': 10, 'accuracies': [], 'names': [], 'expressive': []},
    }
    data_nope = {
        'Star Free': {'marker': 'x', 'marker_size': 15, 'accuracies': [], 'names': [], 'expressive': []},
        'Non Star Free': {'marker': 'x', 'marker_size': 15, 'accuracies': [], 'names': [], 'expressive': []},
    }

    # Define the colormap for the gradient (green to red)
    cmap = mpl.cm.get_cmap('RdYlGn_r')  # Reversed so green is at the lower end and red is at the higher end
    norm = mcolors.Normalize(vmin=1, vmax=7)  # Normalizing for dot_depth values from 1 to 12

    # Populate data for only Star Free and Non Star Free categories
    for data_spec, ape_or_nope in zip([data_ape, data_nope], [ape, nope]):
        for lang, info in ape_or_nope.items():
            category = info['Type']
            name = info['name']
            if category not in data_spec:  # Skip Counter as before
                continue            
            if 'dot_depth' in info:
                # values range from 0, 1, 2, 3, 4 and 12
                if info['dot_depth'] == 12:
                    expressive = 7
                elif info['dot_depth'] > 2:
                    expressive = info['dot_depth'] + 2
                else: 
                    expressive = info['dot_depth']
            else: 
                # Default value for NSF
                expressive = 0
            accuracy = [info['Bin 0'], info['Bin 1'], info['Bin 2']]
            data_spec[category]['accuracies'].append(accuracy)
            data_spec[category]['names'].append(name)
            data_spec[category]['expressive'].append(expressive)

    non_algorithmic_languages = ['Non Star Free', 'Star Free']
    y_limits = (-10, 110)

    # Create a figure and a (3x6) grid of subplots using plt.subplots
    fig, axes = plt.subplots(3, 6, figsize=(20, 8), constrained_layout=True)
    axes = axes.ravel()  # Flatten the axes array to make indexing easier

    # Plot on the right side (3x6 subplots)
    for over_idx, data_ape_or_nope in enumerate([data_ape, data_nope]):
        count = 0
        for category in non_algorithmic_languages:
            for idx, (name, accuracies, expressive) in enumerate(zip(data_ape_or_nope[category]['names'], data_ape_or_nope[category]['accuracies'], data_ape_or_nope[category]['expressive'])):
                ax = axes[count]  # Select the appropriate axis
                
                # Determine the color based on the dot_depth
                if expressive == 0:
                    color = 'grey'  # If dot_depth is 0, use grey
                else:
                    color = cmap(norm(expressive))  # Use colormap for dot_depth from 1 to 12
                
                # Line style - solid for APE; dotted for NOPE
                ax.plot(x, accuracies, marker=data_ape_or_nope[category]['marker'], linestyle=line_styles[over_idx], color=color, markersize=data_ape[category]['marker_size'])

                # Optionally add titles and adjust appearance
                ax.set_title(f'{name}')
                ax.set_xticks(x)
                ax.set_xticklabels(['Bin 1', 'Bin 2', 'Bin 3'])
                ax.set_ylim(y_limits)
                ax.grid(True)
                
                count += 1

    ax = axes[-1]
    ax.axis('off')  # Turn off the last subplot
    
    # Add a legend for the different line styles and dot_depth gradient
    legend_entries = [
        mlines.Line2D([], [], color=cmap(norm(1)), linestyle='solid', label='Dot Depth=1'),
        mlines.Line2D([], [], color=cmap(norm(2)), linestyle='solid', label='Dot Depth=2'),
        mlines.Line2D([], [], color=cmap(norm(3)), linestyle='solid', label='Dot Depth=3'),
        mlines.Line2D([], [], color=cmap(norm(4)), linestyle='solid', label='Dot Depth=4'),
        mlines.Line2D([], [], color=cmap(norm(5)), linestyle='solid', label='Dot Depth=12'),
        # mlines.Line2D([], [], color='black', linestyle='solid', label='APE', marker='>'),
        # mlines.Line2D([], [], color='black', linestyle='dotted', label='NoPE', marker='x')
    ]
    
    # Create a custom legend at the top of the figure, outside the subplots
    fig.legend(handles=legend_entries, loc='upper center', ncol=5, frameon=False)

    # Save or display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reduce the bottom rect to fit the legend at the top
    plt.savefig('dot_depth_vs_others.pdf', bbox_inches="tight")


# Call the function to plot the figure
plot_with_legend()
# dot_depth()
# ac0_vs_lengen()
# ac0_vs_lengen_algo()
# dotdepth_vs_lengen()
