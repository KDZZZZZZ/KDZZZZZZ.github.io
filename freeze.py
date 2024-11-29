from flask_frozen import Freezer
from app import app, Post, User, init_db
import os

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
    # 创建构建目录
    if not os.path.exists('build'):
        os.makedirs('build')
    
    # 确保数据库存在
    if not os.path.exists('blog.db'):
        init_db()
    
    # 生成静态文件
    freezer.freeze()
