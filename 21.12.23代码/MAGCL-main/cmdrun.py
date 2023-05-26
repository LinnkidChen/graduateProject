
import subprocess

for _ in range(1,100):
    try:
        subprocess.run(["python main_acm.py --dataset ACM >>ACMLog.txt"], shell=True ,check=True)
    except subprocess.CalledProcessError as err:
        print("Error running command:", err)