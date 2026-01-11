## Windows PowerShell 示例：推送已有仓库

假设：
- 你在 `E:\资料_临时\SCI\train` 目录下已经 `git init` 过，并且上一条日志提示 `nothing to commit`，说明暂存区和工作区都为空。
- 你确定目录里有代码文件，但未被 `git add`。

### 1. 添加文件并提交
```powershell
# 进入工程目录
cd E:\资料_临时\SCI\train

# 将所有文件添加到 Git 暂存区
git add .

# 查看暂存区状态（可选）
git status

# 提交到本地仓库 master/main 分支
git commit -m "Initial commit"
```

如果再次出现 `nothing to commit`，请确认：
- 目录里确实有文件且不在 `.gitignore`；
- `.git` 文件夹存在。

### 2. 关联远程仓库
```powershell
# 替换下方 URL 为你的 GitHub 仓库地址
$repo = "https://github.com/<USERNAME>/<REPO>.git"

git remote add origin $repo
```
> 若已执行过 `git remote add origin`，再次执行会报错，可先删除再添加：
> `git remote remove origin`

### 3. 推送到 GitHub
```powershell
# 将 master 改为 main（可选）
git branch -M main

# 初次推送，并把 main 与 origin/main 建立追踪关系
git push -u origin main
```
> 首次 push 会要求输入 GitHub 用户名和 **Personal Access Token**。

---

### 常见错误排查
- `fatal: remote origin already exists.`
  - 解决：`git remote remove origin` 然后重新添加。
- `nothing to commit` 但目录有文件：
  - 文件被 `.gitignore` 排除？用 `git add -f` 强制添加。
  - 仍无效，可跑 `git status -u` 查看未跟踪文件。
- 推送时报权限/认证错误：
  - 确保使用 token，token 勾选 `repo` 权限。

