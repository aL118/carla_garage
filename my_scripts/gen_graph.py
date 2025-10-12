import matplotlib.pyplot as plt
import numpy as np
import csv

# Parse CSV file to extract columns 1, 3, and 5 from first table
csv_path = '/fs/nexus-scratch/aliu1237/carla_garage/results/pretrained_longest6/results.csv'

routes = []
ds_means = []
rc_means = []

with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

    # Find the first table (starts after line 19 with "")
    # Header is at line 20, data starts at line 21
    start_idx = 20  # Header row
    end_idx = 31    # Row with "" that ends the table

    for i in range(start_idx, end_idx):
        row = lines[i]
        if len(row) >= 5:
            routes.append(row[0])      # Column 1: route
            ds_means.append(float(row[2]))  # Column 3: DS mean
            rc_means.append(float(row[4]))  # Column 5: RC mean

for means in [ds_means, rc_means]:
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(routes))
    bars = ax.bar(x_pos, means, color='steelblue', edgecolor='black', linewidth=0.7)

    # Customize the chart
    ax.set_xlabel('Route', fontsize=12, fontweight='bold')
    ax.set_ylabel('DC' if means == ds_means else 'RC', fontsize=12, fontweight='bold')
    graph_title = 'Pretrained Driving Score per Route' if means == ds_means else 'Pretrained Completion Percentage per Route'
    ax.set_title(graph_title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(routes, rotation=45, ha='right')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on top of bars
    for i, (bar, value) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    title = 'pretrained_longest6_rc.png' if means == rc_means else 'pretrained_longest6_ds.png'
    plt.savefig(title, dpi=300, bbox_inches='tight')
    print("Bar chart saved")
    plt.show()