"""规则包自洽性：包内互相引用必须走相对导入，任何缩进层级都不许写裸模块名。

规则包的模块会被写进采集器 backend/ 里运行，包内裸名导入只有在脚本式 CLI 下才成立；
放到函数体内（缩进）的裸名导入最容易漏掉，因为导入期不报错，要跑到那行才炸。
"""
import ast
import importlib
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / 'backend' / 'rules'


def test_rules_package_uses_relative_imports_only():
    names = {p.stem for p in PKG.glob('*.py')} - {'__init__'}
    flat = []
    for file in sorted(PKG.glob('*.py')):
        for node in ast.walk(ast.parse(file.read_text(encoding='utf-8'))):
            if isinstance(node, ast.ImportFrom) and not node.level and node.module:
                if node.module.split('.')[0] in names:
                    flat.append('%s:%d from %s' % (file.name, node.lineno, node.module))
            elif isinstance(node, ast.Import):
                flat += ['%s:%d import %s' % (file.name, node.lineno, alias.name)
                         for alias in node.names if alias.name.split('.')[0] in names]
    assert flat == []


def test_rules_package_modules_import():
    for file in sorted(PKG.glob('*.py')):
        if file.stem != '__init__':
            importlib.import_module('backend.rules.%s' % file.stem)
