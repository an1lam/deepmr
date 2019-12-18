#!/usr/bin/python

import os
import subprocess


def main(project_dir):
    subprocess.run(f"isort -s1 {project_dir}/src/*.py", check=True)
    subprocess.run(f"autoflake {project_dir}/src/*.py", check=True)


if __name__ == "__main__":
    project_dir = os.getenv("PROJECT_DIR", default=os.getcwd())
    main(project_dir)
