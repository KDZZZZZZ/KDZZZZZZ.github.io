from app import init_db

if __name__ == '__main__':
    print("Initializing database...")
    import sqlite3
    import os
    from werkzeug.security import generate_password_hash

    # 删除现有的数据库文件
    if os.path.exists('blog.db'):
        os.remove('blog.db')

    # 连接到数据库
    conn = sqlite3.connect('blog.db')
    cursor = conn.cursor()

    # 创建用户表
    cursor.execute('''
    CREATE TABLE user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )
    ''')

    # 创建文章表
    cursor.execute('''
    CREATE TABLE post (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        author_id INTEGER NOT NULL,
        FOREIGN KEY (author_id) REFERENCES user (id)
    )
    ''')

    # 插入管理员用户
    admin_password = generate_password_hash('admin')
    cursor.execute('INSERT INTO user (username, password_hash) VALUES (?, ?)',
                  ('admin', admin_password))

    # 提交更改
    conn.commit()
    conn.close()

    print("Database initialized successfully!")
