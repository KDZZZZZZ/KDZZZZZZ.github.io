from flask_frozen import Freezer
from app import app, Post, User, init_db
import os
import shutil
import sys

# 配置 Freezer
app.config['FREEZER_DESTINATION'] = 'build'
app.config['FREEZER_RELATIVE_URLS'] = True
app.config['FREEZER_BASE_URL'] = 'https://kdzzzzzz.github.io/'

freezer = Freezer(app)

@freezer.register_generator
def index():
    yield {}  # 主页

@freezer.register_generator
def post():
    # 确保数据库存在并有数据
    if not os.path.exists('blog.db'):
        print("数据库不存在，正在初始化...", file=sys.stderr)
        init_db()
    
    # 生成所有文章的 URL
    try:
        posts = Post.query.all()
        print(f"找到 {len(posts)} 篇文章", file=sys.stderr)
        for post in posts:
            yield {'post_id': post.id}
    except Exception as e:
        print(f"获取文章时出错: {str(e)}", file=sys.stderr)
        raise

@freezer.register_generator
def static():
    # 生成静态文件的 URL
    yield {'filename': 'css/style.css'}
    yield {'filename': 'js/main.js'}

if __name__ == '__main__':
    print("开始生成静态文件...", file=sys.stderr)
    
    # 清理并创建构建目录
    if os.path.exists('build'):
        shutil.rmtree('build')
    os.makedirs('build')
    print("构建目录已创建", file=sys.stderr)
    
    # 确保数据库存在
    if not os.path.exists('blog.db'):
        print("初始化数据库...", file=sys.stderr)
        init_db()
    
    # 复制静态文件
    if os.path.exists('static'):
        print("复制静态文件...", file=sys.stderr)
        if os.path.exists('build/static'):
            shutil.rmtree('build/static')
        shutil.copytree('static', 'build/static')
    
    # 生成静态文件
    print("生成页面...", file=sys.stderr)
    try:
        freezer.freeze()
        print("静态文件生成完成！", file=sys.stderr)
    except Exception as e:
        print(f"生成静态文件时出错: {str(e)}", file=sys.stderr)
        sys.exit(1)
