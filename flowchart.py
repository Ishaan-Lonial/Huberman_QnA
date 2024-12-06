from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment="Project Pipeline", format="png")

# Define nodes for each step
steps = {
    "step1": "Download Videos\n(using yt-dlp)",
    "step2": "Transcribe Audio\n(using Whisper)",
    "step3": "Chunk Transcripts\n(for LLM token limit)",
    "step4": "Create Retrieval\nMechanism (TF-IDF)",
    "step5": "Integrate Retrieval\nwith GPT-4",
    "step6": "Fine-tune Prompts\nfor Accuracy",
    "step7": "Streamlit Interface\n(for User Interaction)"
}

# Add nodes
dot.node("step1", steps["step1"], shape="box")
dot.node("step2", steps["step2"], shape="box")
dot.node("step3", steps["step3"], shape="box")
dot.node("step4", steps["step4"], shape="box")
dot.node("step5", steps["step5"], shape="box")
dot.node("step6", steps["step6"], shape="box")
dot.node("step7", steps["step7"], shape="box")

# Add edges to create a 2-2-2-1 layout
dot.edge("step1", "step2")
dot.edge("step2", "step3")
dot.edge("step3", "step4")
dot.edge("step4", "step5")
dot.edge("step5", "step6")
dot.edge("step6", "step7")

# Render the flowchart
output_file = "project_pipeline_flowchart_2_2_2_1"
dot.render(output_file, view=True)

print(f"Flowchart saved as {output_file}.png")
