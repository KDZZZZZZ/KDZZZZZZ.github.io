import os
import shutil

# 定义要清空的文件夹路径
folder_path = 'C:\Users\Administrator\Desktop\blog'

# 删除文件夹中的所有文件
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')

# 重新构建博客
# 假设使用Hugo作为静态网站生成器
os.system('hugo')

# 如果使用其他静态网站生成器，请替换上面的命令
# 例如，如果使用Jekyll，则使用 os.system('jekyll build')