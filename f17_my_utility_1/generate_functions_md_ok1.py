# f17_my_utility_1/generate_functions_md_ok1.py
# Run: python -m f17_my_utility_1.generate_functions_md
# ابتدا باید لیست فایلهای موضوع را در یک فایل تکست قرار بدهیم

import ast
from pathlib import Path

def extract_functions(py_file_path: str):
    """
    Extract top-level function signatures from a Python file.
    """
    path = Path(py_file_path)

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(path))

    functions = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):

            args = []
            defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + node.args.defaults

            for arg, default in zip(node.args.args, defaults):

                arg_name = arg.arg

                annotation = None
                if arg.annotation:
                    annotation = ast.unparse(arg.annotation)

                default_value = None
                if default:
                    default_value = ast.unparse(default)

                if annotation and default_value:
                    arg_sig = f"{arg_name}: {annotation} = {default_value}"
                elif annotation:
                    arg_sig = f"{arg_name}: {annotation}"
                elif default_value:
                    arg_sig = f"{arg_name} = {default_value}"
                else:
                    arg_sig = arg_name

                args.append(arg_sig)

            return_annotation = ""
            if node.returns:
                return_annotation = f" -> {ast.unparse(node.returns)}"

            signature = f"{node.name}({', '.join(args)}){return_annotation}"
            functions.append(signature)

    return functions


def generate_markdown_from_list(file_list_txt: str, output_md: str = "functions_lists.md"):
    """
    Read python file paths from txt and write all function signatures to a markdown file.
    """

    file_list_path = Path(file_list_txt)

    if not file_list_path.exists():
        raise FileNotFoundError(f"File list not found: {file_list_txt}")

    with file_list_path.open("r", encoding="utf-8") as f:
        files = [line.strip() for line in f if line.strip()]

    with open(output_md, "w", encoding="utf-8") as md:

        md.write("# Python Functions List\n\n")

        i = 0
        for py_file in files:
            i += 1
            md.write(f"## File_{i}: `{py_file}`\n\n")

            try:
                functions = extract_functions(py_file)

                if not functions:
                    md.write("_No top-level functions found_\n\n")
                    continue

                for func in functions:
                    md.write(f"- `{func}`\n")

                md.write("\n")

            except Exception as e:
                md.write(f"_Error processing file: {e}_\n\n")

    print(f"Markdown file generated: {output_md}")


if __name__ == "__main__":
    generate_markdown_from_list(
        file_list_txt = "f17_my_utility_1/f03_files_050427.txt",
        output_md     = "f17_my_utility_1/f03_files_050427.md",
    )
    # generate_markdown_from_list(
    #     file_list_txt = "f17_my_utility/f03_indic_files.txt",
    #     output_md     = "f17_my_utility/func_f03_indic.md",
    # )
    # generate_markdown_from_list(
    #     file_list_txt = "f17_my_utility/f03_PrAct_files.txt",
    #     output_md     = "f17_my_utility/func_f03_PrAct.md",
    # )
