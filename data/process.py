import os

cnt = {}

for root, dirs, files in os.walk('./train'):
    for dir in dirs:
        #os.mkdir(os.path.join('./val', dir))
        cnt[dir] = 0
        print dir

for root, dirs, files in os.walk('./train'):
    for file in files:
        d = root.split('/')[-1]
        if not d in cnt:
            continue
        if cnt[d] < 5:
            order = 'mv ' + os.path.join(root, file) + ' '  + os.path.join('./val', d, file)
            print order
            os.system(order)
            cnt[d] += 1
