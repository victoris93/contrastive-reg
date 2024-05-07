from nbconvert import PythonExporter

# Instantiate the PythonExporter
exporter = PythonExporter()

# Read the notebook file
notebook_file = 'graph_rep_2.ipynb'
with open(notebook_file, 'r', encoding='utf-8') as f:
    notebook_content = f.read()

# Convert the notebook to Python code
python_code, _ = exporter.from_notebook_node(notebook_content)

# Write the Python code to a .py file
python_file = 'graph_rep_2.py'
with open(python_file, 'w', encoding='utf-8') as f:
    f.write(python_code)

print(f"Python script saved as '{python_file}'")
