# f17_my_utility_1/generate_imports_md.py
# Run:
#     python -m f17_my_utility_1.generate_imports_md
#
# ابتدا لیست فایل‌های پایتون را در یک فایل txt قرار دهید.
# در هر سطر فقط یک مسیر فایل نوشته شود.
#
# مثال:
# f04_features/core.py
# f04_features/utils.py
# f03_data/data_handler.py

import ast
from pathlib import Path

# -----------------------------------------------------------------------------
# استخراج import ها
# -----------------------------------------------------------------------------
def extract_imports(py_file_path: str) -> list[str]:
    """
    Extract all top-level import statements from a Python module.
    """
    path = Path(py_file_path)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(path))

    imports = []
    for node in tree.body:

        # ---------------------------------------------------------
        # import xxx
        # ---------------------------------------------------------
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    imports.append(f"import {alias.name} as {alias.asname}")
                else:
                    imports.append(f"import {alias.name}")

        # ---------------------------------------------------------
        # from xxx import yyy
        # ---------------------------------------------------------
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            names = []
            for alias in node.names:
                if alias.asname:
                    names.append(f"{alias.name} as {alias.asname}")
                else:
                    names.append(alias.name)
            
            imports.append(f"from {module} import {', '.join(names)}")
    return imports

# -----------------------------------------------------------------------------
# تولید فایل Markdown
# -----------------------------------------------------------------------------
def generate_markdown_from_list(
    file_list_txt: str,
    output_md: str = "imports_list.md",
):
    """
    Read a text file containing Python file paths and generate
    a Markdown report of all import statements.
    """
    file_list_path = Path(file_list_txt)

    if not file_list_path.exists():
        raise FileNotFoundError(file_list_txt)

    with file_list_path.open("r", encoding="utf-8") as f:
        files = [
            line.strip()
            for line in f
            if line.strip()
        ]

    with open(output_md, "w", encoding="utf-8") as md:
        md.write("# Python Import Summary\n\n")

        for i, py_file in enumerate(files, start=1):

            md.write(f"## File_{i}: `{py_file}`\n\n")

            try:
                imports = extract_imports(py_file)
                md.write("### Imports\n\n")
                
                if imports:
                    md.write("```python\n")
                    for item in imports:
                        md.write(item + "\n")
                    md.write("```\n\n")
                else:
                    md.write("_No imports_\n\n")

            except Exception as e:
                md.write(f"_Error processing file: {e}_\n\n")

            md.write("---\n\n")
    print(f"Markdown file generated: {output_md}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    generate_markdown_from_list(
        file_list_txt="f17_my_utility_1/f03_imports_050428.txt",
        output_md="f17_my_utility_1/f03_imports_050428.md",
    )