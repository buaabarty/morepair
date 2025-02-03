import subprocess
import os, glob
import sys
import uuid

def test(id, name, tag):
    time_limit = 60
    tname = str(uuid.uuid4())
    command = "cat ./evalrepair-cpp-res/" + tag + '/fixed' + str(id) + "/" + name + f".cpp ./evalrepair-cpp-res/extend-test/{name}.cpp > tmp/{tname}.cpp && g++ --std=c++17 tmp/{tname}.cpp -lcrypto -o tmp/{tname} && timeout %d ./tmp/{tname}" % time_limit
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print("Standard Output:", stdout.decode('utf-8', 'ignore'))
    print("Standard Error:", stderr.decode('utf-8', 'ignore'))
    print(process.returncode)
    return [process.returncode, str(stdout) + str(stderr)]

cnt = {}
ac1 = {}
ac = {}
ac5 = {}
bug = 0


for id in range(10):
    directory_path = f"./evalrepair-cpp-res/{sys.argv[1]}/fixed{id}/"
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for file_path in sorted(glob.glob(os.path.join(directory_path, '*.cpp')), reverse=False):
            print(file_path)
            name = file_path.split('/')[-1].split('.')[0]
            if sys.argv[-1] == "rejudge":
                ret, detail = test(id, name, sys.argv[1])
            else:
                ret = (int)(open(file_path + '.result', 'r').read())
            if ret == 0:
                ac[name] = ac.get(name, 0) + 1
                if id < 5:
                    ac5[name] = ac5.get(name, 0) + 1
                if id == 0:
                    ac1[name] = ac1.get(name, 0) + 1

print('TOP-10:', len(ac) / 164 * 100, 'TOP-5:', len(ac5) / 164 * 100, 'TOP-1:', len(ac1) / 164 * 100)
