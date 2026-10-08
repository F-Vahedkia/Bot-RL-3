# f17_my_utility/generate_flat_address.py

# Run:
# python -m f17_my_utility_1.generate_flat_address

import os

from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern


def load_gitignore(base_path):
    """
    خواندن .gitignore از ریشه پروژه (اگر وجود داشته باشد)
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


def flat_file_list(
    base_path="e:/Bot-RL-3",
    output_file="f17_my_utility_1/FLAT_ADDRESS.md",
):
    """
    تولید لیست Flat از فایل‌های Python و Markdown پروژه
    با مسیر نسبی نسبت به ریشه پروژه.

    قوانین:
    - .gitignore رعایت می‌شود.
    - فقط فایل‌های .py و .md در نظر گرفته می‌شوند.
    - فایل‌هایی که نامشان با "__" شروع شود نادیده گرفته می‌شوند.
    - پوشه‌ها در خروجی نمایش داده نمی‌شوند.
    - مسیر همه فایل‌ها از ریشه پروژه نوشته می‌شود.

    مثال:

        f08_risk/tests_ch4_i0prev/f08_risk_tester_A.py
    """

    base_path = os.path.abspath(base_path)
    spec = load_gitignore(base_path)

    def is_ignored(path):
        """
        بررسی می‌کند که فایل توسط .gitignore نادیده گرفته شده است یا خیر.
        """
        if not spec:
            return False

        rel_path = os.path.relpath(path, base_path)
        return spec.match_file(rel_path)

    files = []

    for current_path, dirs, filenames in os.walk(base_path):

        # حذف پوشه‌های ignored از ادامه پیمایش
        dirs[:] = [
            d
            for d in dirs
            if not is_ignored(os.path.join(current_path, d))
        ]

        for filename in filenames:

            # فقط فایل‌های Python و Markdown
            if not filename.lower().endswith((".py", ".md")):
                continue

            # نادیده گرفتن فایل‌هایی که نامشان با "__" شروع می‌شود
            if filename.startswith("__"):
                continue

            full_path = os.path.join(current_path, filename)

            # رعایت .gitignore
            if is_ignored(full_path):
                continue

            # مسیر نسبی نسبت به ریشه پروژه
            rel_path = os.path.relpath(full_path, base_path)

            # تبدیل \ به / برای خروجی مناسب
            rel_path = rel_path.replace(os.sep, "/")

            files.append(rel_path)

    # ترتیب deterministic
    files.sort()

    output_path = os.path.abspath(output_file)

    # جلوگیری از اینکه خود فایل خروجی داخل لیست قرار بگیرد
    output_rel_path = os.path.relpath(
        output_path,
        base_path,
    ).replace(os.sep, "/")

    files = [
        path
        for path in files
        if path != output_rel_path
    ]

    # اطمینان از وجود پوشه مقصد
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(files))

    print(f"✅ Flat file list written to {output_file}")


if __name__ == "__main__":

    flat_file_list(
        base_path="e:/Bot-RL-3/",
        output_file="f17_my_utility_1/FLAT_ADDRESS.md",
    )


# Run:
# python -m f17_my_utility_1.generate_flat_address

