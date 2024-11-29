import os
import sys
from datetime import datetime
import sqlite3

def create_new_post(title, content):
    """创建新的博客文章"""
    # 连接数据库
    conn = sqlite3.connect('blog.db')
    cursor = conn.cursor()
    
    # 获取当前时间
    now = datetime.now()
    
    try:
        # 获取管理员用户 ID
        cursor.execute('SELECT id FROM user WHERE username = ?', ('admin',))
        admin_id = cursor.fetchone()[0]
        
        # 插入新文章
        cursor.execute(
            'INSERT INTO post (title, content, created, author_id) VALUES (?, ?, ?, ?)',
            (title, content, now, admin_id)
        )
        conn.commit()
        print(f"\n✅ 博客《{title}》发布成功！")
        
    except Exception as e:
        print(f"\n❌ 发布失败: {str(e)}", file=sys.stderr)
        conn.rollback()
        
    finally:
        conn.close()

def read_file_content(file_path):
    """从文件读取内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"\n❌ 读取文件失败: {str(e)}", file=sys.stderr)
        return None

def main():
    print("\n=== 发布新博客 ===")
    
    # 检查是否提供了文件路径
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        content = read_file_content(file_path)
        if content is None:
            return
        
        # 使用文件名作为标题（去掉扩展名）
        title = os.path.splitext(os.path.basename(file_path))[0]
        
    else:
        # 获取标题
        title = input("\n请输入博客标题: ").strip()
        if not title:
            print("\n❌ 错误：标题不能为空")
            return
        
        # 获取内容
        print("\n请输入博客内容 (输入 'EOF' 单独一行来结束):")
        content_lines = []
        while True:
            line = input()
            if line.strip() == 'EOF':
                break
            content_lines.append(line)
        
        content = '\n'.join(content_lines)
    
    if not content:
        print("\n❌ 错误：内容不能为空")
        return
    
    # 创建新文章
    create_new_post(title, content)
    
    # 提示部署
    print("\n要立即部署到网站吗？")
    print("1. 运行 python deploy.py 来部署")
    print("2. 或者等待推送到 GitHub 后自动部署")

if __name__ == '__main__':
    main()
