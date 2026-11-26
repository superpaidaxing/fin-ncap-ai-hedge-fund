# Fin-NCAP 金融实战评测与源码二开实录（Exam C v1.0）

本项目是基于 [`virattt/ai-hedge-fund`](https://github.com/virattt/ai-hedge-fund)（v2.7.2，MIT，基线标签 `fin-ncap-baseline`）在 **Fin-NCAP 考卷 C（上市公司财报尽调与财务排雷魔鬼考卷 v1.0）** 上开展实测与联合源码二开的公开可复现仓库。

## 一、核心改造清单（3 个原子 Commit，可直接 `git log fin-ncap-baseline..HEAD` 查看完整 Diff）

1. **`feat(forensic)` — 确定性财务排雷与三表勾稽门禁 (`hedge_fund/forensic/`)**：
   - 落实 **"A Prompt Is Not a Control"** 原则：资产负债表勾稽不平、非标审计意见、超限违规对外担保触发 **Fail-Closed 一票否决**；
   - 10 项纯 Python 确定性排雷探针（存贷双高、隐含货币资金收益率倒挂、受限/归集存款、净现比背离、连续负 FCF、应收 DSO、存货膨胀[含高毛利零负债陈酿白马防误杀保护]、商誉/有形净资产、无形资产激进资本化、少数股东藏亏、控股股东高质押）；
   - 内置免收费 API 的通用标准三表 JSON 适配器 `LocalFinancialStatementClient`。
2. **`fix(snapshot)` — 透传三表排雷审计并修复盲测负 FCF 消失 Bug (`hedge_fund/features/snapshot.py`)**：
   - 修复原版 `_per_share_index` 在最早一期 `FCF <= 0` 时导致盲测模式 `fcf_idx` 整列变成 `-` 的源码缺陷，显式输出 `FCF sign audit`；
   - 将确定性排雷比率、有形净资产与红旗清单结构化注入大师 Prompt。
3. **`feat(signals)` — 确定性风控天花板与截断 JSON 恢复 (`hedge_fund/signals/llm_agent.py`)**：
   - 触发 `FAIL_CLOSED` 时直接短路拒绝（0 次 LLM 调用）；
   - 触发严重红旗 `BEARISH_CAP` 时强制封顶最终信号，禁止 LLM 越权对雷股喊多，并保留 `llm_raw_signal` 供审计对比；
   - 修复截断 JSON 导致怀疑票静默弃权的问题。

## 二、一键复现与 A 股实战尽调命令

```bash
# 1. 运行全部单元测试（含原版 + 确定性排雷 + Tushare A 股适配器测试）
pytest -q hedge_fund/data/test_tushare_client.py hedge_fund/forensic/ hedge_fund/features/test_snapshot.py hedge_fund/signals/test_llm_agents.py

# 2. 查看全部源码 Diff（原版基线 -> A 股实战与确定性排雷二开版）
git diff fin-ncap-baseline..HEAD -- hedge_fund/

# 3. 零 API Key 离线验证考卷 C 确定性排雷门禁
python examples/exam_c_repro/verify_forensic_gate.py

# 4. 真实 A 股实战尽调（需配置 TUSHARE_TOKEN 与任意大模型 Key，支持中文 5 大师全上）
aihf analyze 600519.SH                               # 实时体检最新年报（默认中文 + 5 位大师）
aihf analyze 600518.SH --as-of 2018-05-05 --no-forensic  # 时光机对照：原版 12 列视角
aihf analyze 600518.SH --as-of 2018-05-05                # 时光机对照：二开确定性排雷门禁 ON
```

## 三、诚实边界声明

本二开工具定位为**“财报初筛排雷与财务勾稽硬护栏”**，无法解决：① 上游财报 PDF 解析误差；② 表外空壳闭环造假与未披露抽屉担保；③ 银行/保险/地产/强周期等特殊行业需动态校准阈值；④ 不能替代持牌人员终审或直接接管实盘交易。
