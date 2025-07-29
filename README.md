
## PowerShellの実行エラー時
<!-- セキュリティ保護の一環として「実行ポリシー」があり、スクリプトの実行が制限されることがある -->
Get-ExecutionPolicy -List　<-- 実行ポリシーの確認
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned　<-- 一時的にスクリプト許可

## 環境構築

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Git管理
git init
echo venv/ > .gitignore
git add .
git commit -m "Initial commit"

