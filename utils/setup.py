import os
import sys


print("##"*30)
print("Install libraries")
os.system(f"pip install -r requirements.txt")
print("##"*30)
print("Login wandb")
if len(sys.argv) > 1:
    os.system(f"wandb login {sys.argv[1]}")
else:
    print("Not logged in to wandb yet!")
