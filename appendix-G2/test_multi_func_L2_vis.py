import matplotlib.pyplot as plt

# # Data from results_d32 and results_d256
# results_d32 = {"j=i-c": {"c=1": [0.0124985492, 496.2664794922, 755.2360839844,], 
#                          "c=5": [0.0271074250, 333.3888244629, 606.5682983398,], 
#                          "c=25": [0.0000022057, 0.7152619362, 1.0706132650,]},
#                 "j>i-c": {"c=2": [0.0083255535, 303.7139282227, 414.6399841309,],
#                           "c=5": [0.0255450439, 216.1955566406, 381.6783447266,],
#                           "c=25": [0.0078058084, 9.3571901321, 15.1211357117,]},
#                 "(i-j)=c_2 mod c_1": {"c_1=3, c_2=0": [0.0000034845, 0.0010645119, 0.0053957077,],
#                                       "c_1=5, c_2=0": [0.0000032848, 0.0002877587, 0.0015938719,],
#                                       "c_1=3, c_2=2": [0.0000013159, 0.0000026057, 0.0000058817,],
#                                       "c_1=5, c_2=2": [0.0000028226, 0.0000027044, 0.0000029815,],}
# }

# results_d256 = {"j=i-c": {"c=1": [0.0000045789, 0.2099193931, 0.3275869489,], 
#                          "c=5": [0.0000039742, 0.2041597813, 0.3235522509,], 
#                          "c=25": [0.0000049315, 0.1646348983, 0.2828634381,]},
#                 "j>i-c": {"c=2": [0.0000372034, 0.3495599926, 0.5547776222, ],
#                           "c=5": [0.0001623197, 0.6075839400, 1.0019609928, ],
#                           "c=25": [0.0000049899, 1.8029826880, 3.1008999348, ]},
#                 "(i-j)=c_2 mod c_1": {"c_1=3, c_2=0": [0.0003711144, 2.8137290478, 6.8732824326,],
#                                       "c_1=5, c_2=0": [0.0003158526, 2.3210136890, 5.1842579842,],
#                                       "c_1=3, c_2=2": [0.0006015065, 3.1011548042, 7.2998561859,],
#                                       "c_1=5, c_2=2": [0.0000395538, 1.9786781073, 4.6821579933,],}
# }


# # Function to plot the data
# def plot_data(results_d32, results_d256):
#     fig, axs = plt.subplots(2, 3, figsize=(15, 8))
#     # fig.suptitle("MSE Loss on Different Lengths", fontsize=16)

#     # Define the range for symlog
#     linthresh = 1  # Threshold for switching between linear and log
#     y_min, y_max = -0.3, 1e3  # Range of y-axis

#     # LaTeX labels
#     latex_labels = {
#         "j=i-c": r"$\phi$: j = i - c",
#         "j>i-c": r"$\phi$: j > i - c",
#         "(i-j)=c_2 mod c_1": r"$\phi$: (i - j) = $c_2$ mod $c_1$"
#     }

#     # Plot settings
#     def plot_subplot(ax, data, title):
#         for key, values in data.items():
#             ax.plot(values, label=f"${key}$")
#         ax.set_title(title)
#         ax.set_yscale('symlog', linthresh=linthresh)  # Symmetric log scale
#         ax.set_ylim([y_min, y_max])  # Set the same y-axis range for all plots
#         ax.set_xticks([0, 1, 2])  # Positions for x-ticks (for 3 values)
#         ax.set_xticklabels([50, 100, 150])  # Set custom x-tick labels
#         ax.set_ylabel('Loss')  # Y-axis label
#         ax.set_xlabel('Length')  # X-axis label
#         ax.legend()

#     # Row 1: results_d32
#     plot_subplot(axs[0, 0], results_d32["j=i-c"], f"{latex_labels['j=i-c']} ; d=32")
#     plot_subplot(axs[0, 1], results_d32["j>i-c"], f"{latex_labels['j>i-c']} ; d=32")
#     plot_subplot(axs[0, 2], results_d32["(i-j)=c_2 mod c_1"], f"{latex_labels['(i-j)=c_2 mod c_1']} ; d=32")

#     # Row 2: results_d256
#     plot_subplot(axs[1, 0], results_d256["j=i-c"], f"{latex_labels['j=i-c']} ; d=256")
#     plot_subplot(axs[1, 1], results_d256["j>i-c"], f"{latex_labels['j>i-c']} ; d=256")
#     plot_subplot(axs[1, 2], results_d256["(i-j)=c_2 mod c_1"], f"{latex_labels['(i-j)=c_2 mod c_1']} ; d=256")

#     # Fine-tuning layout
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.92)  # Adjust to leave space for the main title
#     plt.show()

# # Plot the figures
# plot_data(results_d32, results_d256)




# Data from results_d32 and results_d256
results_d32 = {"j<i-c": {"c=0": [0.0708710179, 229.7775573730, 368.5939636230,],
                          "c=5": [0.0573513247, 95.5051727295, 212.6814727783,],
                          "c=25": [0.0000146985, 9.1086015701, 13.8165292740,]},
                "i-j is prime": {" ": [0.0521395244, 7.8534722328, 17.9720077515,],},
                "combined": {"j=i-1 | (i-j)=0 mod 3": [0.0234131571, 234.0101318359, 420.2164001465,],
                             "j=i-10 | (i-j)=0 mod 3": [0.1561257094, 4.5674924850, 17.3307113647,],
                             "j=i-1 | (i-j)=0 mod 5": [0.0284165759, 160.3611602783, 290.1488037109,],
                             "j=i-12 | (i-j)=0 mod 5": [0.1483774185, 2.0993382931, 5.5044059753,],}
}

results_d256 = {"j<i-c": {"c=0": [0.0003972818, 5.5996322632, 16.4334640503,],
                          "c=5": [0.0002407077, 9.2956085205, 21.4885139465,],
                          "c=25": [0.0000289284, 14.2473096848, 26.0415172577,]},
                "i-j is prime": {" ": [0.0000668037, 3.6927046776, 7.2145252228,],},
                "combined": {"j=i-1 | (i-j)=0 mod 3": [0.0000087227, 3.2345695496, 7.5253567696, ],
                             "j=i-10 | (i-j)=0 mod 3": [0.0002122433, 3.2348296642, 7.5272789001, ],
                             "j=i-1 | (i-j)=0 mod 5": [0.0000581382, 2.7456986904, 5.7526345253,],
                             "j=i-12 | (i-j)=0 mod 5": [0.0000177166, 2.7219662666, 5.7598857880, ],}
}


# Function to plot the data
def plot_data(results_d32, results_d256):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    # fig.suptitle("MSE Loss on Different Lengths", fontsize=16)

    # Define the range for symlog
    linthresh = 1  # Threshold for switching between linear and log
    y_min, y_max = -0.3, 1e3  # Range of y-axis

    # LaTeX labels
    latex_labels = {
        "j<i-c": r"$\phi$: j < i - c",
        "i-j is prime": r"$\phi$: i-j is prime",
        "combined": r"combined"
    }

    # Plot settings
    def plot_subplot(ax, data, title):
        for key, values in data.items():
            ax.plot(values, label=f"{key}")
        ax.set_title(title)
        ax.set_yscale('symlog', linthresh=linthresh)  # Symmetric log scale
        ax.set_ylim([y_min, y_max])  # Set the same y-axis range for all plots
        ax.set_xticks([0, 1, 2])  # Positions for x-ticks (for 3 values)
        ax.set_xticklabels([50, 100, 150])  # Set custom x-tick labels
        ax.set_ylabel('Loss')  # Y-axis label
        ax.set_xlabel('Length')  # X-axis label
        if key != " ":
            ax.legend()

    # Row 1: results_d32
    plot_subplot(axs[0, 0], results_d32["j<i-c"], f"{latex_labels['j<i-c']} ; d=32")
    plot_subplot(axs[0, 1], results_d32["i-j is prime"], f"{latex_labels['i-j is prime']} ; d=32")
    plot_subplot(axs[0, 2], results_d32["combined"], f"{latex_labels['combined']} ; d=32")

    # Row 2: results_d256
    plot_subplot(axs[1, 0], results_d256["j<i-c"], f"{latex_labels['j<i-c']} ; d=256")
    plot_subplot(axs[1, 1], results_d256["i-j is prime"], f"{latex_labels['i-j is prime']} ; d=256")
    plot_subplot(axs[1, 2], results_d256["combined"], f"{latex_labels['combined']} ; d=256")

    # Fine-tuning layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust to leave space for the main title
    plt.show()

# Plot the figures
plot_data(results_d32, results_d256)