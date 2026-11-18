# 机械硬盘存储

目标硬盘：D:（WSL `/mnt/d`），WDC WD40 PURX-78AKYY0，4 TB HDD。

迁移范围以研究工作区的本地 `hdd_large_files.json` 为准，共34个不小于128 MiB的文件，约158.4 GiB。迁移清单和进度文件均保存在本地。小文件、代码、最终图件、论文及共享运行时图谱索引保留在固态。

- 真实大文件：`/mnt/d/ASPR-data/`，沿用项目相对目录结构。
- 项目原路径：迁移成功后为符号链接，已有代码和证据路径继续有效。
- 状态：`docs/hdd_migration.json`，只有 `complete` 表示已校验并释放本地文件。
- 复制中的 `.partial` 不用于分析，原文件在校验完成前保留。
- 校验：源文件复制时计算SHA-256，目标完整读取复核；同时检查源文件大小及修改时间。
- 续跑入口：`python3 scripts/maintenance/migrate_bulk_to_hdd.py`（不要在现有迁移进程运行时重复启动）。

机械硬盘需要保持挂载；拔出后相关链接不可访问。Windows WSL虚拟磁盘内部释放空间与C盘VHDX实际缩小是两回事，本脚本不关闭WSL或压缩VHDX。
