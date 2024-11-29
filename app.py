from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
import markdown2
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///blog.db')
if app.config['SQLALCHEMY_DATABASE_URI'].startswith('postgres://'):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# 用户模型
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    posts = db.relationship('Post', backref='author', lazy=True)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

# 博客文章模型
class Post(db.Model):
    __tablename__ = 'post'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    summary = db.Column(db.String(500))
    image_url = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f'<Post {self.title}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    page = request.args.get('page', 1, type=int)
    posts = Post.query.order_by(Post.created_at.desc()).all()  # 先检查是否有文章
    print(f"找到 {len(posts)} 篇文章")  # 添加调试信息
    posts_page = Post.query.order_by(Post.created_at.desc()).paginate(page=page, per_page=5)
    return render_template('home.html', posts=posts_page)

@app.route('/post/<int:post_id>')
def post(post_id):
    post = Post.query.get_or_404(post_id)
    content = markdown2.markdown(post.content)
    return render_template('post.html', post=post, content=content)

@app.route('/post/<int:post_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        flash('您没有权限编辑这篇文章')
        return redirect(url_for('post', post_id=post.id))
    
    if request.method == 'POST':
        post.title = request.form['title']
        post.content = request.form['content']
        post.summary = request.form.get('summary', '')
        post.image_url = request.form.get('image_url', '')
        
        db.session.commit()
        flash('文章已更新', 'success')
        return redirect(url_for('post', post_id=post.id))
    
    return render_template('edit_post.html', post=post)

@app.route('/post/<int:post_id>/delete', methods=['POST'])
@login_required
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        flash('您没有权限删除这篇文章')
        return redirect(url_for('post', post_id=post.id))
    
    db.session.delete(post)
    db.session.commit()
    flash('文章已删除', 'success')
    return redirect(url_for('home'))

@app.route('/create', methods=['GET', 'POST'])
@login_required
def create_post():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        summary = request.form.get('summary', '')
        image_url = request.form.get('image_url', '')
        
        post = Post(
            title=title,
            content=content,
            summary=summary,
            image_url=image_url,
            author=current_user
        )
        
        db.session.add(post)
        db.session.commit()
        flash('文章发布成功！', 'success')
        return redirect(url_for('home'))
    
    return render_template('create_post.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):
            login_user(user)
            return redirect(url_for('home'))
        flash('用户名或密码错误')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

def init_db():
    print("正在初始化数据库...")
    with app.app_context():
        db.drop_all()  # 删除所有表
        db.create_all()  # 重新创建表
        print("数据库表已创建")
        
        # 检查是否已经有用户
        if not User.query.filter_by(username='admin').first():
            print("创建管理员用户...")
            admin = User(username='admin')
            admin.set_password('admin')
            db.session.add(admin)
            db.session.commit()
            print("管理员用户已创建")
            
            # 添加示例文章
            print("添加示例文章...")
            try:
                with open('example_post.md', 'r', encoding='utf-8') as f:
                    content = f.read()
                    post = Post(
                        title='欢迎使用个人博客系统！',
                        content=content,
                        summary='这是一个示例文章，展示了 Markdown 的基本用法和博客系统的主要功能。',
                        image_url='https://picsum.photos/800/400',
                        author=admin,
                        created_at=datetime.utcnow()
                    )
                    db.session.add(post)
                    db.session.commit()
                    print("示例文章已添加")
            except Exception as e:
                print(f"添加示例文章时出错：{e}")
                db.session.rollback()
        else:
            print("管理员用户已存在，跳过初始化")

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
