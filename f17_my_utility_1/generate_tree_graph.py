# f17_my_utility/generate_tree_graph.py
# Run: python -m f17_my_utility_1.generate_tree_graph

import os
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern


def load_gitignore(base_path):
    """
    خواندن .gitignore از پوشه مقصد (اگر وجود داشته باشد)
    """
    gitignore_path = os.path.join(base_path, ".gitignore")
    if not os.path.isfile(gitignore_path):
        return None

    with open(gitignore_path, "r", encoding="utf-8") as f:
        patterns = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    return PathSpec.from_lines(GitWildMatchPattern, patterns)


def tree_md(base_path="e:/Bot-RL-3", output_file="f17_my_utility_1/files_list.md"):
    """
    تولید نمودار درختی Markdown با رعایت .gitignore
    و نمایش پوشه‌ها قبل از فایل‌ها
    """
    spec = load_gitignore(base_path)
    base_path = os.path.abspath(base_path)

    def is_ignored(path):
        if not spec:
            return False
        rel_path = os.path.relpath(path, base_path)
        return spec.match_file(rel_path)

    def walk(current_path, prefix=""):
        entries = os.listdir(current_path)

        # جداسازی پوشه‌ها و فایل‌ها
        dirs = []
        files = []

        for e in entries:
            full_path = os.path.join(current_path, e)
            if is_ignored(full_path):
                continue
            if os.path.isdir(full_path):
                dirs.append(e)
            else:
                files.append(e)

        dirs.sort()
        files.sort()
        items = dirs + files

        lines = []
        for idx, name in enumerate(items):
            full_path = os.path.join(current_path, name)
            is_last = idx == len(items) - 1
            connector = "└── " if is_last else "├── "

            if os.path.isdir(full_path):
                lines.append(f"{prefix}{connector}**{name}/**")
                extension = "    " if is_last else "│   "
                lines.extend(walk(full_path, prefix + extension))
            else:
                lines.append(f"{prefix}{connector}{name}")

        return lines

    lines = walk(base_path)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Tree written to {output_file}")


if __name__ == "__main__":
    # folder_path = input("Path to folder: ").strip()
    # tree_md(folder_path)
    tree_md(
        base_path  = "e:/Bot-RL-3/",
        output_file = "f17_my_utility_1/TREE_GRAPH_1.md",
    )

# Run: python -m f17_my_utility_1.generate_tree_graph