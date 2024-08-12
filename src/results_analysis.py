# src/results_analysis.py

import pandas as pd

def wait_for_compute_plan(client, compute_plan_key):
    print("Waiting for compute plan to complete...")
    client.wait_compute_plan(compute_plan_key)
    print("Compute plan completed.")

def get_performance_table(client, compute_plan_key):
    performances = client.get_performances(compute_plan_key).model_dump()
    df = pd.DataFrame(performances)
    return df[["worker", "round_idx", "identifier", "performance"]]

def print_performance_table(df):
    print("\nPerformance Table:")
    print(df.to_string(index=False))

def analyze_results(client, compute_plan_key):
    wait_for_compute_plan(client, compute_plan_key)
    performance_df = get_performance_table(client, compute_plan_key)
    print_performance_table(performance_df)
    
    # You can add more analysis functions here
    # For example:
    # plot_performance_over_rounds(performance_df)
    # calculate_average_performance(performance_df)