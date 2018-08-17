"""

随机获得20个测试样本的序号
"""

import random

file = list(range(130))

test_list = []
for _ in range(20):
    temp = random.choice(file)
    file.remove(temp)

    test_list.append(temp)

test_list.sort()

for item in test_list:
    print(item)

# 2
# 20
# 27
# 29
# 30
# 37
# 52
# 57
# 62
# 63
# 68
# 69
# 70
# 71
# 79
# 97
# 111
# 114
# 117
# 123
