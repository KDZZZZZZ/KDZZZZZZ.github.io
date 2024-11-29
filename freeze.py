from flask_frozen import Freezer
from app import app, Post, User, init_db
import os
import shutil

# 配置 Freezer
app.config['FREEZER_DESTINATION'] = 'build'
app.config['FREEZER_RELATIVE_URLS'] = True
freezer = Freezer(app)

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
