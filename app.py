import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

final_results_df = pd.read_csv('processed.csv')

# Columns to choose from
columns = [col for col in final_results_df.columns if col not in ['PDB_File', 'Residue']]

# Streamlit App
st.title('PDB Files Analysis Dashboard')

# Column selection
selected_column = st.selectbox('Select a column to visualize its distribution:', columns)

# Plot distribution
if selected_column:
    fig = px.histogram(
        final_results_df,
        x=selected_column,
        nbins=50,
        title=f'Distribution of {selected_column} (normalized)',
        histnorm='probability density'  # Normalize the histogram
    )
    st.plotly_chart(fig)






# # Numeric columns to choose from
# numeric_columns = [
#     col for col in final_results_df.columns
#     if col not in ['PDB_File', 'Residue'] and pd.api.types.is_numeric_dtype(final_results_df[col])
# ]

# def plot_mean_with_confidence_intervals(df, column, show_mean=True, show_conf_interval=True):
#     grouped = df.groupby('PDB_File')[column]
#     means = grouped.mean()
#     sems = grouped.sem()  # Standard error of the mean

#     fig = go.Figure()

#     if show_mean:
#         # Add mean line
#         fig.add_trace(go.Scatter(
#             x=means.index,
#             y=means.values,
#             mode='lines+markers',
#             name='Mean',
#             line=dict(color='blue')
#         ))

#     if show_conf_interval:
#         # Add confidence interval (mean ± SEM)
#         fig.add_trace(go.Scatter(
#             x=means.index,
#             y=means.values + sems.values,
#             fill=None,
#             mode='lines',
#             line=dict(color='lightblue'),
#             showlegend=False
#         ))

#         fig.add_trace(go.Scatter(
#             x=means.index,
#             y=means.values - sems.values,
#             fill='tonexty',  # Fill the area between this trace and the previous one
#             mode='lines',
#             line=dict(color='lightblue'),
#             name='Confidence Interval'
#         ))

#     fig.update_layout(
#         title=f'Mean and Confidence Interval of {column} across PDB Files',
#         xaxis_title='PDB File',
#         yaxis_title=column
#     )

#     return fig

# # Streamlit app layout
# st.title('PDB Files Analysis')
# st.sidebar.header('Settings')

# # Select column to visualize
# selected_column = st.sidebar.selectbox('Select column to visualize', numeric_columns)

# # Toggle mean and confidence interval
# show_mean = st.sidebar.checkbox('Show Mean', value=True)
# show_conf_interval = st.sidebar.checkbox('Show Confidence Interval', value=True)

# # Plot the selected column
# st.header(f'Visualization of {selected_column}')
# fig = plot_mean_with_confidence_intervals(final_results_df, selected_column, show_mean, show_conf_interval)
# st.plotly_chart(fig)

# # Display the original dataframe
# st.header('Original DataFrame')
# st.dataframe(final_results_df)