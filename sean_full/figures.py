import plotly.graph_objects as go

labels = [
    "Task 1", "Neuron Selection T1", "Training T1", "Storing Mask", "Check Capacity", "Grow Network",
    "Task 2", "Reuse 50% T1 Neurons", "Select 50% New Neurons", "Training T2 (EWC Loss)",
    "Task 3", "Reuse 50% T2 Neurons", "Select 50% New Neurons (T3)", "Training T3 (EWC Loss)",
]

sources = [
    0, 1, 2, 2, 4,
    6, 7, 8, 9, 9, 4,
    10, 11, 12, 13, 13, 4
]

targets = [
    1, 2, 3, 4, 5,
    7, 8, 9, 3, 4, 5,
    11, 12, 13, 3, 4, 5
]

values = [1] * len(sources)

link_colors = [
    "rgba(31, 119, 180, 0.6)"] * 5 + \
    ["rgba(255, 127, 14, 0.6)"] * 6 + \
    ["rgba(44, 160, 44, 0.6)"] * 6

fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        pad=20,
        thickness=30,
        line=dict(color="black", width=0.5),
        label=labels,
        color="lightblue"
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=link_colors
    )
)])

fig.update_layout(
    title_text="Continual Learning Methodology Pipeline",
    font=dict(size=20, color='black'),
    height=600
)

fig.show()
