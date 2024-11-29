# 博客使用指南

## 发布新博客

### 方法一：使用脚本发布（推荐）

1. 运行发布脚本：
```bash
python new_post.py
```

2. 按提示输入博客信息：
   - 输入博客标题
   - 输入博客内容（支持 Markdown 格式）
   - 输入 'EOF' 单独一行来结束内容输入

示例：
```bash
$ python new_post.py

=== 发布新博客 ===

请输入博客标题: 我的第一篇博客

请输入博客内容 (输入 'EOF' 单独一行来结束):
# 你好，世界！

这是我的第一篇博客。

## 关于我
我是一个热爱技术的开发者。

EOF

✅ 博客《我的第一篇博客》发布成功！
```

### 方法二：直接操作数据库

如果你熟悉 SQLite，也可以直接向数据库添加文章：

```sql
INSERT INTO post (title, content, created, author_id) 
VALUES ('文章标题', '文章内容', CURRENT_TIMESTAMP, 1);
```

## 部署博客

### 方法一：自动部署（推荐）

1. 提交更改：
```bash
git add .
git commit -m "发布新博客：文章标题"
```

2. 推送到 GitHub：
```bash
git push origin main
```

3. GitHub Actions 会自动构建并部署你的博客

### 方法二：手动部署

如果你想立即看到更改，可以使用手动部署：

```bash
python deploy.py
```

## 本地预览

1. 启动开发服务器：
```bash
python app.py
```

2. 在浏览器中访问：
```
http://localhost:5000
```

## 编写技巧

### Markdown 支持

博客内容支持 Markdown 格式，你可以使用：

- # 一级标题
- ## 二级标题
- **粗体**
- *斜体*
- [链接](URL)
- ![图片](图片URL)
- 代码块：
  ```python
  print("Hello, World!")
  ```

### 文章管理

- 所有文章都存储在 SQLite 数据库中
- 可以通过 Web 界面编辑或删除文章
- 支持按时间排序和分页显示

## 故障排除

### 部署失败

1. 检查 GitHub Actions 状态
2. 确保所有依赖都已安装：
```bash
pip install -r requirements.txt
```

### 数据库问题

如果遇到数据库问题，可以重新初始化：
```bash
python init_db.py
```

## 备份

建议定期备份 `blog.db` 文件，它包含了所有的博客文章数据。

## 更新日志

### v1.0.0
- 支持 Markdown 格式
- 自动部署到 GitHub Pages
- 本地预览功能
- 文章管理系统
