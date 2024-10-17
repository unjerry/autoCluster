import getopt
import sys
 
# 解析命令行参数
args, values = getopt.getopt(sys.argv[1:], "")
 
# 处理解析结果
for value in values:
    print("位置参数:", value)