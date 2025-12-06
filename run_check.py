import subprocess

with open("check_output_py.txt", "w", encoding="utf-8") as f:
    subprocess.run(["cargo", "check", "-p", "nn", "--tests"], stdout=f, stderr=subprocess.STDOUT)
