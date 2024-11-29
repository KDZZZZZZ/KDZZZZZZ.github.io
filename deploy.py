import os
import subprocess
import sys

def run_command(command, cwd=None):
    """运行命令并打印输出"""
    print(f"\n>>> Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result

def deploy():
    """部署博客到 GitHub Pages"""
    # 1. 确保我们在主分支上
    run_command(['git', 'checkout', 'main'])
    
    # 2. 生成静态文件
    print("\n>>> Generating static files...")
    import freeze
    
    # 3. 保存当前分支的更改
    run_command(['git', 'add', '.'])
    try:
        run_command(['git', 'commit', '-m', 'Update content'])
    except:
        print("No changes to commit on main branch")
    
    # 4. 切换到 gh-pages 分支
    print("\n>>> Switching to gh-pages branch...")
    try:
        run_command(['git', 'checkout', 'gh-pages'])
    except:
        run_command(['git', 'checkout', '--orphan', 'gh-pages'])
    
    # 5. 清理工作目录
    print("\n>>> Cleaning working directory...")
    for item in os.listdir('.'):
        if item != '.git' and item != 'build':
            if os.path.isfile(item):
                os.remove(item)
            else:
                import shutil
                shutil.rmtree(item)
    
    # 6. 移动构建文件到根目录
    print("\n>>> Moving build files...")
    for item in os.listdir('build'):
        src = os.path.join('build', item)
        if os.path.isfile(src):
            os.rename(src, item)
        else:
            if os.path.exists(item):
                shutil.rmtree(item)
            shutil.move(src, item)
    os.rmdir('build')
    
    # 7. 提交更改
    print("\n>>> Committing changes...")
    run_command(['git', 'add', '.'])
    try:
        run_command(['git', 'commit', '-m', 'Deploy to GitHub Pages'])
    except:
        print("No changes to deploy")
    
    # 8. 推送到 GitHub
    print("\n>>> Pushing to GitHub...")
    run_command(['git', 'push', 'origin', 'gh-pages', '--force'])
    
    # 9. 切回主分支
    print("\n>>> Switching back to main branch...")
    run_command(['git', 'checkout', 'main'])
    
    print("\n>>> Deployment complete! Your site should be live in a few minutes.")

if __name__ == '__main__':
    deploy()
