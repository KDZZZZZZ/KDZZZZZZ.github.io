# 个人博客系统

一个使用 Flask 和 Tailwind CSS 构建的现代化个人博客系统。

## 功能特点

- 🚀 现代化的响应式设计
- 📝 Markdown 文章支持
- 🖼️ 文章封面图片
- 📱 移动端适配
- 🔐 用户认证系统
- 📄 文章分页
- 🌐 中文界面

## 技术栈

- 后端框架：Flask 2.3.3
- 数据库：SQLite + SQLAlchemy
- 前端样式：Tailwind CSS
- Markdown 支持：markdown2
- 用户认证：Flask-Login

## 快速开始

1. 克隆仓库：
```bash
git clone https://github.com/你的用户名/blog.git
cd blog
```

2. 创建虚拟环境：
```bash
python -m venv venv
```

3. 激活虚拟环境：
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

4. 安装依赖：
```bash
pip install -r requirements.txt
```

5. 运行应用：
```bash
python app.py
```

6. 访问网站：
打开浏览器访问 http://localhost:5000

## 默认管理员账户

- 用户名：admin
- 密码：admin

首次运行时会自动创建管理员账户和示例文章。

## 目录结构

```
blog/
├── app.py              # 主应用程序
├── requirements.txt    # 项目依赖
├── example_post.md    # 示例文章
├── static/            # 静态文件
│   ├── css/          # 样式文件
│   └── js/           # JavaScript 文件
└── templates/         # 模板文件
    ├── base.html     # 基础模板
    ├── home.html     # 首页模板
    ├── post.html     # 文章页模板
    ├── login.html    # 登录页模板
    └── edit_post.html # 编辑文章模板
```

## 主要功能

1. 文章管理
   - 创建新文章
   - 编辑已有文章
   - 删除文章
   - Markdown 格式支持
   - 文章摘要
   - 封面图片

2. 用户系统
   - 用户登录
   - 权限控制
   - 安全密码存储

3. 界面设计
   - 响应式布局
   - 现代化 UI
   - 优雅的排版

## 开发计划

- [ ] 添加文章分类功能
- [ ] 实现标签系统
- [ ] 添加搜索功能
- [ ] 评论系统
- [ ] 用户注册功能
- [ ] 文件上传功能
- [ ] 社交媒体分享
- [ ] 文章统计
- [ ] SEO 优化

## 贡献指南

1. Fork 本仓库
2. 创建新分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'Add some feature'`
4. 推送分支：`git push origin feature/your-feature`
5. 提交 Pull Request

## 许可证

MIT License
