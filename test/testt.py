import getopt
import sys

# 定义短选项和长选项
short_options = "ho:v"
long_options = ["help", "output=", "verbose"]

# 解析命令行参数
args, values = getopt.getopt(sys.argv[1:], short_options, long_options)

# 遍历解析结果
print(args)
print(values)
for opt, arg in args:
    if opt in ("-h", "--help"):
        print("显示帮助信息")
    elif opt in ("-o", "--output"):
        print("输出文件路径：", arg)
    elif opt == "-v":
        print("启用详细输出")
