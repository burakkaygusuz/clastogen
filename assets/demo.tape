# Makes assets/demo.gif. Run from the repository root: vhs assets/demo.tape
Output assets/demo.gif

Require uv

Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 650
Set Padding 20
Set Margin 20
Set MarginFill "#1e1e2e"
Set WindowBar Colorful
Set BorderRadius 8
Set TypingSpeed 40ms
Set Theme "Catppuccin Mocha"

Hide
Type "export PS1='$ ' && uv sync --quiet && clear"
Enter
Sleep 5s
Show

# 1. The eval suite is green.
Type "uv run pytest -q examples/test_banking_eval.py"
Sleep 500ms
Enter
Sleep 3s

# 2. Clastogen finds that the suite misses 2 of 5 prompt faults, and the build fails.
Type "uv run pytest --clastogen --clastogen-fail-under=80 -q examples/test_banking_eval.py"
Sleep 500ms
Enter
Sleep 10s
