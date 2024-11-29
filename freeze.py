from flask_frozen import Freezer
from app import app, Post, User, init_db
import os
import shutil

# 配置 Freezer
app.config['FREEZER_DESTINATION'] = 'build'
app.config['FREEZER_RELATIVE_URLS'] = True
app.config['PREFERRED_URL_SCHEME'] = 'https'
freezer = Freezer(app)

@freezer.register_generator
def index():
    # 生成分页页面的 URL
    posts = Post.query.count()
    pages = (posts - 1) // 5 + 1
    for page in range(1, pages + 1):
        yield {'page': page}

@freezer.register_generator
def post():
    # 确保数据库存在并有数据
    if not os.path.exists('blog.db'):
        init_db()
    
    # 生成所有文章的 URL
    for post in Post.query.all():
        yield {'post_id': post.id}

if __name__ == '__main__':
    # 清理并创建构建目录
    if os.path.exists('build'):
        shutil.rmtree('build')
    os.makedirs('build')
    
    # 确保数据库存在
    if not os.path.exists('blog.db'):
        init_db()
    
    # 复制静态文件
    if os.path.exists('static'):
        shutil.copytree('static', 'build/static')
    
    # 生成静态文件
    freezer.freeze()
