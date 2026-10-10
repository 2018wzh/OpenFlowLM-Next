# OpenFlowLM capture API v1

所有客户端使用 `/api/v1`。SSE 事件含 `schema_version: 1`、`protocol: "openflowlm.capture.v1"`、`run_id` 和单调递增的 `event_seq`。旧的无版本 HTTP 接口已移除。原生二进制文件与事件索引仍采用 capture format 1。

## 启动与能力

```sh
python utilities/moe_capture/serve.py --allow-origin http://localhost:5173
```

服务默认只监听本机。额外前端源通过可重复的 `--allow-origin` 显式允许，包括 SSE 和 OPTIONS 预检；前端以服务地址为 API base URL。默认 context capacity 来自内核 manifest 的 `max_ctx_default`，可通过 `--max-ctx` 调整后重启 runtime。容量是实际分配的 KV/状态容量，不能靠请求参数越界。请求体默认最多 16 MiB，可用 `--max-request-bytes` 调整。API 可独立启动，无需预先构建此仓库网页；已有网页产物则同时提供静态页面。

`GET /api/v1/capabilities` 返回模型 family、层数、层类型、hidden、experts、TopK、vocab、context capacity、manifest 导出能力、采集范围、队列及传输选项。客户端必须以此决定控件范围。当前原生引擎支持 Qwen3.6-35B-A3B 的完整 40 层；支持适配前端，不代表支持任意模型架构。

`GET /api/v1/schema` 返回带容量范围的 JSON Schema 请求定义。`POST /api/v1/tokenize` 接收 `{text,focus?}`，返回实际 input_ids、带字符 span 的 token、默认采集位置和 target_position；无需等待模型加载，其他前端可先对齐位置再提交采集。

`GET /api/v1/status` 返回模型状态 `loading | ready | running | error`、最近提交的 run ID 和其状态。

## 提交

`POST /api/v1/runs`，Content-Type 为 application/json；成功返回 HTTP 202 与 `run_id`。模型加载或执行过程中也可提交，按 FIFO 排队。每次执行完整重置模型状态，不做多请求并发 NPU 执行。

```json
{
  "max_tokens": 16,
  "compare": true,
  "cases": [{
    "id": "river",
    "text": "He waited beside the river bank",
    "focus": "bank",
    "capture_positions": [5],
    "target_position": 5,
    "vector_layers": [1, 7, 39],
    "micro_layers": [1, 7, 39],
    "micro_rank": 2,
    "ablate_layer": 7,
    "ablation_ranks": [2, 5]
  }]
}
```

- `cases` 是非空数组，无两案例限制。`id` 可省略，自动生成 live_N；同一请求内须唯一。
- `text` 使用模型包 tokenizer 编码，不加 special tokens。`focus` 可省略，默认最后一个输入 token；提供时选择最后一次出现的文本所对应的全部 token，target 默认其中最后一个。
- `capture_positions` 可为实际输入位置数组或 `"all"`；默认 focus 对齐的位置。`target_position` 必须在这些位置中。
- `vector_layers` 可为 0–39 数组或 `"all"`，默认 `[0,3]`。所有层均可导出归一化 MoE 输入、Top8 原始输出、routed mixture、shared 输出和门值、前后残差；路由始终采集全部层及输入/续写位置。
- `micro_layers` 可为数组或 `"all"`，后者表示全部已选 vector layers，默认 `[3]`。必须是 vector_layers 的子集。每个选定位置/层导出 `micro_rank` 对应专家的 gate/up/SiLU/product/BF16 down input/down，rank 范围 0–7，默认 0。
- `ablate_layer` 可为任意层，默认 3。`ablation_ranks` 可为任意互不重复的 Top8 排名数组，默认 `[0,7]`，空数组仅比较 baseline。干预点是 target_position × ablate_layer，专家身份由该点 baseline 冻结，保持原路由权重且不重新归一化。
- `max_tokens` 默认 8，可由每个 case 覆盖；无 8/256 token 上限，受实际 context capacity 限制。输入与生成或固定续写的最大长度之和必须不超过容量。
- case 可提供 `continuation` 文本；compare 为 true 时按该固定续写计算，否则使用 baseline 生成的续写。
- `compare` 默认 false。true 执行 baseline teacher forcing 及指定排名消融。rank 0/7 的运行名是 target/control，其余是 ablate_rank_N。microscope 在干预点冻结专家身份，其他点按当次实际排名选择。

未知请求字段或无效索引报错，不静默忽略、不替换为默认值。

## 运行管理

- `GET /api/v1/runs?offset=0&limit=100` 分页列出历史，含 total 与 next_offset。
- `GET /api/v1/runs/<id>` 返回状态、时间与原始请求，不返回本机路径。
- `POST /api/v1/runs/<id>/cancel` 取消排队任务；执行中的任务在下一个 token 边界退出，模型继续驻留。
- 状态为 queued/running/cancelling/complete/cancelled/error/incomplete。取消或失败的部分数据保留，但没有正常 complete 事件，不可当完整数据使用。
- 每次运行独立存储，重启恢复历史；上次未结束的任务标 incomplete，不自动重放。

## 事件与张量

`GET /api/v1/runs/<id>/events?tensors=inline` 返回 SSE；`tensors=references` 只返回描述符。客户端可以使用 `Last-Event-ID` 或 `after=<seq>` 从已接收位置续接，前者优先。事件类别包括 metadata/case/run/route/moe_input/moe_output/expert_detail/logits/continuation/comparison/complete。未知扩展类别应忽略；未知协议版本应报错。失败或取消使用 `event: failure`，客户端关闭 EventSource 并根据运行状态展示部分结果。

张量描述符含 name/dtype/shape/offset/length/source/event_seq，以及专家张量的 slot/expert_id。dtype 为 f32/i32/bf16，字节序 little。inline 模式的非 logits 张量附带 Base64 `data`；所有张量均提供相对 `href`。完整词表 logits 始终通过 href 按需读取，以避免每个 SSE 包携带大型词表。logits 事件另附 derived Top16 概率与 next_token_id 对应的文本。

`GET /api/v1/runs/<id>/tensors/<event_seq>/<tensor_index>` 返回已发布张量的原始 application/octet-stream 字节。发布前已刷新二进制数据；未发布张量返回 404。保留 BF16 uint16 原始位模式；显示浮点时将每个 uint16 左移 16 位，再按 float32 解释。每个事件中的 tensor_index 按 tensors 数组排序，offset 是原始连续文件偏移，不是下载响应内偏移。

原生张量 source 为 native_openflowlm；概率、似然及前端加权乘积等为 derived。精度来源于 metadata/manifest。host_elapsed_ms 表示此连接送达事件时相对任务开始的时间，历史重放也按当次送达计算，不是原始算子时间戳。

```js
const base = 'http://127.0.0.1:4180';
const caps = await fetch(`${base}/api/v1/capabilities`).then(r => r.json());
const response = await fetch(`${base}/api/v1/runs`, {
  method: 'POST', headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({cases: [{text: 'A different prompt',
    vector_layers: [7], micro_layers: [7]}], max_tokens: 16})
});
if (!response.ok) throw new Error(await response.text());
const {run_id} = await response.json();
const events = new EventSource(`${base}/api/v1/runs/${run_id}/events?tensors=references`);
events.onmessage = async ({data}) => {
  const event = JSON.parse(data);
  if (event.schema_version !== 1) { events.close(); throw new Error('Unsupported protocol'); }
  for (const tensor of event.tensors ?? []) {
    const bytes = await fetch(base + tensor.href).then(r => r.arrayBuffer());
    // 按 tensor.dtype 和 shape 解码，保持原生位模式。
  }
  if (event.kind === 'complete') events.close();
};
events.addEventListener('failure', () => events.close());
```

单个 NPU runtime 仍串行运行；这是当前原生执行模型。扩展请求范围会增加推理时间、张量体积和内存需求，API 不通过隐藏截断限制请求规模。
