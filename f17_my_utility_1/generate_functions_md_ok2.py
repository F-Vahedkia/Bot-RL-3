# f17_my_utility_1/generate_functions_md_ok2.py
# Run: python -m f17_my_utility_1.generate_functions_md_ok2
# ابتدا باید لیست فایلهای موضوع را در یک فایل تکست قرار بدهیم


import ast
from pathlib import Path

# =============================================================================
# Signatures
# =============================================================================
def build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """
    Build a readable function/method signature.
    """

    args = []

    defaults = (
        [None] * (len(node.args.args) - len(node.args.defaults))
        + node.args.defaults
    )

    for arg, default in zip(node.args.args, defaults):

        arg_name = arg.arg

        annotation = (
            ast.unparse(arg.annotation)
            if arg.annotation
            else None
        )

        default_value = (
            ast.unparse(default)
            if default
            else None
        )

        if annotation and default_value:
            args.append(f"{arg_name}: {annotation} = {default_value}")
        elif annotation:
            args.append(f"{arg_name}: {annotation}")
        elif default_value:
            args.append(f"{arg_name} = {default_value}")
        else:
            args.append(arg_name)

    # *args
    if node.args.vararg:
        arg = node.args.vararg
        if arg.annotation:
            args.append(f"*{arg.arg}: {ast.unparse(arg.annotation)}")
        else:
            args.append(f"*{arg.arg}")

    # keyword-only args
    if node.args.kwonlyargs:
        if node.args.vararg is None:
            args.append("*")

        for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):

            annotation = (
                ast.unparse(arg.annotation)
                if arg.annotation
                else None
            )

            default_value = (
                ast.unparse(default)
                if default
                else None
            )

            if annotation and default_value:
                args.append(f"{arg.arg}: {annotation} = {default_value}")
            elif annotation:
                args.append(f"{arg.arg}: {annotation}")
            elif default_value:
                args.append(f"{arg.arg} = {default_value}")
            else:
                args.append(arg.arg)

    # **kwargs
    if node.args.kwarg:
        arg = node.args.kwarg
        if arg.annotation:
            args.append(f"**{arg.arg}: {ast.unparse(arg.annotation)}")
        else:
            args.append(f"**{arg.arg}")

    returns = ""
    if node.returns:
        returns = f" -> {ast.unparse(node.returns)}"

    return f"{node.name}({', '.join(args)}){returns}"


# =============================================================================
# Top-Levels
# =============================================================================
def extract_module_members(py_file_path: str):
    """
    Extract top-level functions and classes (with methods).
    """

    path = Path(py_file_path)

    if not path.exists():
        return {
            "functions": [],
            "classes": []
        }

    with path.open("r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(path))

    result = {
        "functions": [],
        "classes": []
    }

    for node in tree.body:
        # -----------------------------
        # Top-level functions
        # -----------------------------
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result["functions"].append(
                build_signature(node)
            )
        # -----------------------------
        # Top-level classes
        # -----------------------------
        elif isinstance(node, ast.ClassDef):
            cls = {
                "name": node.name,
                "methods": []
            }
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cls["methods"].append(
                        build_signature(item)
                    )
            result["classes"].append(cls)
    return result

# =============================================================================
# Generate result file
# =============================================================================
def generate_markdown_from_list(
    file_list_txt: str,
    output_md: str = "functions_lists.md"
):
    """
    Read python file paths and generate markdown.
    """

    file_list_path = Path(file_list_txt)

    if not file_list_path.exists():
        raise FileNotFoundError(file_list_txt)

    with file_list_path.open(
        "r",
        encoding="utf-8"
    ) as f:
        files = [
            line.strip()
            for line in f
            if line.strip()
        ]

    with open(
        output_md,
        "w",
        encoding="utf-8"
    ) as md:

        md.write("# Python Module Summary\n\n")
        for i, py_file in enumerate(files, start=1):
            md.write(f"## File_{i}: `{py_file}`\n\n")

            try:
                data = extract_module_members(py_file)

                # -----------------------------
                # Functions
                # -----------------------------
                md.write("### Top-level Functions\n\n")

                if data["functions"]:
                    for func in data["functions"]:
                        md.write(f"- `{func}`\n")
                else:
                    md.write("_None_\n")

                md.write("\n")

                # -----------------------------
                # Classes
                # -----------------------------
                md.write("### Top-level Classes\n\n")

                if data["classes"]:
                    for cls in data["classes"]:
                        md.write(f"#### class `{cls['name']}`\n\n")
                        if cls["methods"]:
                            for method in cls["methods"]:
                                md.write(f"- `{method}`\n")
                        else:
                            md.write("_No methods_\n")
                        
                        md.write("\n")

                else:
                    md.write("_None_\n\n")

            except Exception as e:
                md.write(f"_Error processing file: {e}_\n\n")

    print(f"Markdown file generated: {output_md}")

# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":

    generate_markdown_from_list(
        file_list_txt="f17_my_utility_1/f03_files_050430.txt",
        output_md="f17_my_utility_1/f03_files_050430.md",
    )