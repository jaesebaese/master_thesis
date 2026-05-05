from IPython.display import Image, display
from supervisor_agent import agent
display(Image(agent.get_graph().draw_mermaid_png()))