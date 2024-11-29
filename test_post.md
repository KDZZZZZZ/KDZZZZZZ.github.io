# Markdown 样式测试

这是一篇测试文章，用来展示各种 Markdown 样式效果。

## 1. 文本样式

普通文本段落。

**粗体文本** 和 *斜体文本*。

~~删除线文本~~ 和 `行内代码`。

## 2. 列表

### 无序列表
- 项目 1
- 项目 2
  - 子项目 2.1
  - 子项目 2.2
- 项目 3

### 有序列表
1. 第一步
2. 第二步
3. 第三步

### 任务列表
- [x] 已完成任务
- [ ] 未完成任务
- [ ] 待办事项

## 3. 引用

> 这是一段引用文本。
> 
> 可以有多行。
> - 也可以包含列表
> - 和其他元素

## 4. 代码块

行内代码：`print("Hello, World!")`

Python 代码块：
```python
def hello_world():
    print("Hello, World!")
    return True

# 带注释的代码
class Example:
    def __init__(self):
        self.value = 42
```

JavaScript 代码块：
```javascript
function calculateSum(a, b) {
    return a + b;
}

// ES6 箭头函数
const multiply = (x, y) => x * y;
```

## 5. 表格

| 功能 | 描述 | 支持状态 |
|------|------|----------|
| Markdown | 基础语法 | ✅ |
| 代码高亮 | 支持多种语言 | ✅ |
| 数学公式 | LaTeX 语法 | ❌ |

## 6. 链接和图片

[访问 GitHub](https://github.com)

![示例图片](https://picsum.photos/800/400)

## 7. 水平分割线

---

## 8. HTML 标签支持

<details>
<summary>点击展开详情</summary>

这是隐藏的内容。
- 可以包含列表
- 和其他 Markdown 元素
</details>

## 9. 数学公式

行内公式：$E = mc^2$

独立公式：
$$
\frac{n!}{k!(n-k)!} = \binom{n}{k}
$$

## 10. 总结

这篇文章展示了：
1. 基础文本格式
2. 列表和任务列表
3. 代码高亮
4. 表格样式
5. 其他 Markdown 元素
