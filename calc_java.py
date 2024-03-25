import subprocess
import os, glob
import sys

def test(id, name, tag):
    command = "cd evalrepair-java && bash test.sh " + str(id) + " " + name + " " + tag
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print("Standard Output:", stdout.decode())
    print("Standard Error:", stderr.decode())
    print(process.returncode)
    return [process.returncode, str(stdout)]

ac = {}
ac5 = {}
bug = 0

ss = 0

for id in range(10):
    directory_path = f"./evalrepair-java-res/{sys.argv[1]}/fixed{id}/"
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for file_path in sorted(glob.glob(os.path.join(directory_path, '*.java')), reverse=False):
            print(file_path)
            name = file_path.split('/')[-1].split('.')[0]
            ret, detail = test(id, name, sys.argv[1])
            res = (int)(open(file_path + '.result', 'r').read())
            if ret == 0:
                ac[name] = ac.get(name, 0) + 1
                if id < 5:
                    ac5[name] = ac5.get(name, 0) + 1
            ss += 1
            print(bug, ss)
            print('TOP-10:', len(ac) / 163 * 100, 'TOP-5:', len(ac5) / 163 * 100, 'TOT:', 163)