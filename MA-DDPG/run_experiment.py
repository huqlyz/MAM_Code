import os

envlist = ["Ant-v2", "Walker2d-v2", "HalfCheetah-v2", "Hopper-v2", "Humanoid-v2", "Reacher-v2",
           "InvertedDoublePendulum-v2", "InvertedPendulum-v2"]
for env_name in envlist:
    for i in range(10):
        str = ('python main.py  --env' + envname + " --seed " + str(i))
        p = os.system(str1)
