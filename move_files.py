import os
import shutil

def move_files_from_build():
    build_dir = 'build'
    # 移动文件
    for root, dirs, files in os.walk(build_dir):
        for file in files:
            src_path = os.path.join(root, file)
            # 计算目标路径（去掉 'build/' 前缀）
            dst_path = os.path.join(*(src_path.split(os.sep)[1:]))
            # 如果目标路径有目录部分，确保目录存在
            dst_dir = os.path.dirname(dst_path)
            if dst_dir:  # 只有当目标路径包含目录时才创建
                os.makedirs(dst_dir, exist_ok=True)
            # 移动文件
            shutil.move(src_path, dst_path)
            print(f'Moved {src_path} to {dst_path}')
    
    # 删除空的 build 目录
    try:
        shutil.rmtree(build_dir)
        print(f'Removed {build_dir} directory')
    except Exception as e:
        print(f'Error removing {build_dir} directory: {e}')

if __name__ == '__main__':
    move_files_from_build()
