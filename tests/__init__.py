# -*- coding: utf-8 -*-
"""xmetai-evaluation 测试包。

分层：
- ``unit/``        纯逻辑与组件级用例（合成数据，毫秒级）
- ``integration/`` 全链用例（合成默认跑；``realdata`` 档需环境变量指向真实数据）
- ``conftest.py``  外层共享 fixture
- ``runner.py``    合成数据工厂 + 评测运行器 + 产物读取（unit / integration 共用）
"""
