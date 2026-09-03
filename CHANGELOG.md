# 变更日志

## 未发布

### 修复

- **多进程实时重建索引容错（H1/H2）**：在 `embedding_meta` 中新增跨进程 `schema_epoch` 计数器。任一进程执行 `rebuildVecTables()` 时递增该计数器；其他存活进程在每次向量写入前通过 `vecEpochChanged()` 比对磁盘上的 epoch 与本进程绑定的 `boundEpoch`。若检测到向量表在本进程脚下被重建（其预处理的 INSERT 语句已绑定到被 DROP 的表），则**跳过向量写入**（元数据与 FTS 仍照常落盘），并冻结向量检索直至重新初始化——被跳过的行会在下一次重建索引时补齐嵌入。跳过永远是安全分支：不会写入失败，也不会写入错误宽度的向量。
- **空库维度不匹配不再每次启动告警（H3）**：恢复“无内容可重嵌入 → 不置 needsReindex、不逐次启动告警”的原有正确行为。当 L1=0 且 L0=0 时，直接按目标维度重建空的向量表，不触发重建索引流程。
- **锁窃取原子化（M1）**：`stealStaleLock` 改用 temp + `link(2)` 原子模式取代非原子的 truncate-write-read，杜绝两个窃取者同时“认为自己赢了”的竞态。
- **锁心跳刷新，TTL 仅作兜底（M2）**：重建索引期间每 30s 刷新锁的心跳 `ts`；陈旧判定阈值放宽至 120s，确保永不从一个存活的重建索引进程手中窃取锁。
- **`scripts/reindex-now.ts` 补齐 D3 就绪等待（M4）**：在执行破坏性重建前，与后台重建索引路径一致地调用 `startWarmup()` 并等待 `isReady()`——避免本地 provider 下“先 DROP 表、再 embed 失败留空”的问题。
- **测试覆盖（P1）**：新增 `src/core/store/reindex.test.ts`（vitest，tmp-dir SQLite + mock embedFn，无网络），覆盖锁状态机（获取/窃取/释放，含并发窃取）、审批门（批准/拒绝）、reindexPending 冻结（跳过向量写入、检索返回空、不崩溃）、`reindexAll` 诚实计数、`markEmbeddingCurrent` 仅在成功后写入。

### 升级说明

- **旧版数据库（无 `embedding_meta`）**：启动时会打印一次告警，且**向量检索保持冻结**（关键词/FTS 检索不受影响，旧向量原样保留在磁盘上），直至你显式批准重建——在内存配置中设置 `"reindex": { "approveChanges": true }` 并重启，或运行 `scripts/reindex-now.ts`。回退嵌入配置即可无损恢复原索引（重建前不销毁任何数据）。

### IMemoryStore 兼容性说明（P3）

- `reindexAll(embedFn, onProgress?)` 的 `onProgress` 回调签名为 `(succeeded, failed, total, layer: "L1" | "L0")`，返回值为每层 `{ l1Count, l0Count, l1Failed, l0Failed, l1Total, l0Total }`。外部 `IMemoryStore` 实现者需与此签名保持一致。
- 新增可选方法 `rebuildVecTables(providerInfo)` 与 `markEmbeddingCurrent(providerInfo)`：破坏性的 drop+recreate 只在审批后的重建路径中发生，绝不在存储初始化时触发。
