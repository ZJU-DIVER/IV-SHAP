import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def plot_boxs(all_data,name,rho):
    # Generate an array containing four sets of random numbers, each set containing 100 data points with standard deviations of 1, 2, 3, 4
    #all_data = [np.random.normal(std, std/10, size=3) for std in range(1, 17)]  # Add a dataset with a standard deviation of 4
    positions_new_data = np.array([0.85,0.95,1.05,1.15,1.85,1.95,2.05,2.15,2.85,2.95,3.05,3.15,3.85,3.95,4.05,4.15]) 
    labels = ['0.125','0.25','0.375','0.5']  # Add corresponding labels

    # print(len(all_data))
    # print(len(labels))
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot notched box plots
    bplot_notch = ax.boxplot(
        x=all_data,         # Data to be plotted
        positions=positions_new_data,
        notch=True,         # Whether to have a notch, default is False
        widths=0.1,         # Box width
        #labels=labels,      # Labels for the box plots
        patch_artist=True,  # Whether to fill the boxes with color, default is False
        medianprops={       # Set properties for the median line
            'linestyle': '-', 'color': 'black', 'linewidth': 1.5
        },
        showfliers=False,   # Whether to show outliers, default is True
        flierprops={        # Set properties for outliers
            'marker': '^', 'markersize': 6.75, 'markeredgewidth': 0.75, 'markerfacecolor': '#ee5500', 'markeredgecolor': 'k'
        },
        whiskerprops={      # Set properties for the whiskers
            'linestyle': '--', 'linewidth': 1.2, 'color': '#480656'
        },
        capprops={          # Set properties for the caps at the ends of whiskers
            'linestyle': '-', 'linewidth': 1.5, 'color': '#480656'
        }
    )

    ax.set_xticks([1.0,2.0,3.0,4.0])  # Set 1.0 as the tick
    ax.set_xticklabels(['0.125','0.25','0.375','0.5'])  # Set labels at position 1.0

    # Fill the box plots with color
    colors = ['lightcoral', 'red',  'lightskyblue','royalblue', 'lightcoral', 'red',  'lightskyblue', 'royalblue', 'lightcoral', 'red',  'lightskyblue', 'royalblue', 'lightcoral', 'red',  'lightskyblue','royalblue']  # Add color yellow
    for patch, color in zip(bplot_notch['boxes'], colors):
        patch.set_facecolor(color)

    colors = ['lightcoral', 'red',  'lightskyblue','royalblue']
    algorithms = ['IV-BSHAP', 'BSHAP', 'IV-IG', 'IG']  # Replace with your actual algorithm names

    # Create patches
    patches = [mpatches.Patch(color=color, label=algorithm) for color, algorithm in zip(colors, algorithms)]
    # Add legend
    plt.legend(handles=patches)
    # Add horizontal grid lines and set axis labels
    ax.yaxis.grid(False)
    title_notch = ax.set_title('ρ={}'.format(rho),weight='bold')
    ax.set_xlabel("Feature Deviation", weight='bold')
    ax.set_ylabel("Error", weight='bold')
    plt.savefig('./pictures/{}.eps'.format(name))
    plt.show()  # Display the figure

def plot_boxs_XGBRegressor(all_data,name,rho):
    # Generate an array containing four sets of random numbers, each set containing 100 data points with standard deviations of 1, 2, 3, 4
    #all_data = [np.random.normal(std, std/10, size=3) for std in range(1, 17)]  # Add a dataset with a standard deviation of 4
    positions_new_data = np.array([0.95,1.05,1.95,2.05,2.95,3.05,3.95,4.05]) 
    labels = ['0.125','0.25','0.375','0.5']  # Add corresponding labels

    # print(len(all_data))
    # print(len(labels))
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot notched box plots
    bplot_notch = ax.boxplot(
        x=all_data,         # Data to be plotted
        positions=positions_new_data,
        notch=True,         # Whether to have a notch, default is False
        widths=0.1,         # Box width
        #labels=labels,      # Labels for the box plots
        patch_artist=True,  # Whether to fill the boxes with color, default is False
        medianprops={       # Set properties for the median line
            'linestyle': '-', 'color': 'black', 'linewidth': 1.5
        },
        showfliers=False,   # Whether to show outliers, default is True
        flierprops={        # Set properties for outliers
            'marker': '^', 'markersize': 6.75, 'markeredgewidth': 0.75, 'markerfacecolor': '#ee5500', 'markeredgecolor': 'k'
        },
        whiskerprops={      # Set properties for the whiskers
            'linestyle': '--', 'linewidth': 1.2, 'color': '#480656'
        },
        capprops={          # Set properties for the caps at the ends of whiskers
            'linestyle': '-', 'linewidth': 1.5, 'color': '#480656'
        }
    )

    ax.set_xticks([1.0,2.0,3.0,4.0])  # Set 1.0 as the tick
    ax.set_xticklabels(['0.125','0.25','0.375','0.5'])  # Set labels at position 1.0

    # Fill the box plots with color
    colors = ['lightcoral', 'red', 'lightcoral', 'red', 'lightcoral', 'red', 'lightcoral', 'red']  # Add color yellow
    for patch, color in zip(bplot_notch['boxes'], colors):
        patch.set_facecolor(color)

    colors = ['lightcoral', 'red']
    algorithms = ['IV-BSHAP', 'SHAP']  # Replace with your actual algorithm names

    # Create patches
    patches = [mpatches.Patch(color=color, label=algorithm) for color, algorithm in zip(colors, algorithms)]
    # Add legend
    plt.legend(handles=patches)
    # Add horizontal grid lines and set axis labels
    ax.yaxis.grid(False)
    title_notch = ax.set_title('ρ={}'.format(rho),weight='bold',fontsize=15)
    ax.set_xlabel("Feature Deviation", weight='bold',fontsize=13)
    ax.set_ylabel("Error", weight='bold',fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=11)
    plt.savefig('./pictures/new_{}.eps'.format(name))
    plt.show()  # Display the figure
