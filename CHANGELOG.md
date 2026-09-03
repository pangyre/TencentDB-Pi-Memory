# 变更日志

## 未发布

### 修复

- **多进程实时重建索引容错（H1/H2）**：在 `embedding_meta` 中新增跨进程 `schema_epoch` 计数器。任一进程执行 `rebuildVecTables()` 时递增该计数器；其他存活进程在每次向量写入前通过 `vecEpochChanged()` 比对磁盘上的 epoch 与本进程绑定的 `boundEpoch`。若检测到向量表在本进程脚下被重建（其预处理的 INSERT 语句已绑定到被 DROP 的表），则**跳过向量写入**（元数据与 FTS 仍照常落盘），并冻结向量检索直至重新初始化——被跳过的行会在下一次重建索引时补齐嵌入。跳过永远是安全分支：不会写入失败，也不会写入错误宽度的向量。
- **空库维度不匹配不再每次启动告警（H3）**：恢复“无内容可重嵌入 → 不置 needsReindex、不逐次启动告警”的原有正确行为。当 L1=0 且 L0=0 时，直接按目标维度重建空的向量表，不触发重建索引流程。
- **重建锁原子化并提取为独立模块（M1）**：新增 `src/core/store/reindex-lock.ts`（`ReindexLock`），tdai-core 与测试共用同一实现。获取采用 唯一临时文件 + `link(2)` 原子抢占——`link` 失败 EEXIST 即放弃，两个进程不可能同时持有锁（取代原先 unlink-then-create 的 TOCTOU 窗口）。持有者以随机 nonce 标识：释放/抢占只操作自己的锁，绝不覆盖后继者的锁。
- **锁心跳 + TTL 兜底（M2）**：重建期间每 60s 心跳刷新锁 `ts`；超过 30 分钟（`maxAgeMs`，可配）的锁视为陈旧、可被抢占——覆盖 pid 复用与跨主机 pid 命名空间下存活检测失效的场景。正常运行远短于该阈值；存活期间心跳保证不会被误抢。
- **`scripts/reindex-now.ts` 补齐 D3 就绪等待（M4）**：在执行破坏性重建前，与后台重建索引路径一致地调用 `startWarmup()` 并等待 `isReady()`——避免本地 provider 下“先 DROP 表、再 embed 失败留空”的问题。
- **测试覆盖（P1）**：新增 `src/core/store/reindex.test.ts`（vitest，tmp-dir SQLite + sqlite-vec + mock embedFn，无网络），覆盖：锁状态机（首获/拒绝/陈旧抢占/nonce 保护释放/心跳续期——直接测试 `ReindexLock` 生产模块）、审批门（批准/拒绝）、reindexPending 冻结（跳过向量写入、检索返回空、不崩溃；以裸 SQL 断言向量行确被跳过）、`reindexAll` 诚实计数（含全部失败场景）、`markEmbeddingCurrent` 仅在全量成功后写入、D1 崩溃安全（未完成的重建在下一次启动重新被检测）。已知测试边界：多进程真并发由 `link(2)` 原子性保证并已在文档说明，单线程测试无法复现该竞态。

### 升级说明

- **旧版数据库（无 `embedding_meta`）**：启动时会打印一次告警，且**向量检索保持冻结**（关键词/FTS 检索不受影响，旧向量原样保留在磁盘上），直至你显式批准重建——在内存配置中设置 `"reindex": { "approveChanges": true }` 并重启，或运行 `scripts/reindex-now.ts`。回退嵌入配置即可无损恢复原索引（重建前不销毁任何数据）。

### IMemoryStore 兼容性说明（P3）

- `reindexAll(embedFn, onProgress?)` 的 `onProgress` 回调签名为 `(succeeded, failed, total, layer: "L1" | "L0")`，返回值为每层 `{ l1Count, l0Count, l1Failed, l0Failed, l1Total, l0Total }`。外部 `IMemoryStore` 实现者需与此签名保持一致。
- 新增可选方法 `rebuildVecTables(providerInfo)` 与 `markEmbeddingCurrent(providerInfo)`：破坏性的 drop+recreate 只在审批后的重建路径中发生，绝不在存储初始化时触发。
