# GPT Image 图像生成

GPT Image 图片生成插件，支持所有 GPT Image 系列模型。

支持手动使用 `/生图`、`/改图` 调用，也支持 LLM 通过函数工具自主调用。兼容 OpenAI 官方 Images API、标准 OpenAI-compatible 接口，以及网页逆向/2api/ToAPIs 渠道；不需要手动告诉插件供应商类型，填好 API Key、Base URL 和模型名即可。

- `/生图`：文生图。
- `/改图`：图生图/图片编辑，自动读取当前消息图片和回复/引用消息里的图片。
- 或者自然语言描述 模型会自动调用工具进行生成

## 配置

在 AstrBot 插件配置里只需要填一组 API：

- `api.api_key`：供应商提供的 API Key。留空时会按环境变量回退读取，默认 `OPENAI_API_KEY` 在前；如果你之前是 2api/ToAPIs 用户(老配置里 `two_api.enabled` 开着)，会优先 `TWO_API_KEY`。
- `api.base_url`：填到 `/v1` 即可，例如 `https://api.openai.com/v1`。插件会自动补 `/images/generations` 或 `/images/edits`。
- `api.model`：OpenAI 官方接口建议使用 `gpt-image-1.5`、`gpt-image-1` 或 `gpt-image-1-mini`；中转/网页逆向渠道按供应商要求填写，例如 `gpt-image-2`、`gpt-image-2-all`、`gpt-image-1.5-official`。
- `api.timeout_seconds`：图像生成可能较慢，建议 120-300 秒。
- `api.user_agent`：可选。留空时使用插件默认 UA（避开 Cloudflare 对 Python/aiohttp 默认 UA 的拦截规则）；只有当中转明确报 `HTTP 403 cf-ray=...` 或要求特定 UA 时才需要在这里覆盖。

插件内部维护两套参数档案 (`standard` 适配 OpenAI 标准 Images API；`flexible` 适配网页逆向/2api/ToAPIs)。一次请求会先按启发式或上次成功的档案下单；如果上游因为参数格式问题报错，再用另一套档案自动重试一次，成功的档案会缓存到下一次。所以无论你的供应商是哪一种格式都不用手动切换。

(PS:网页逆向渠道一般只能生成 1K 分辨率，如果你不清楚自己的模型是哪种类型可以参考中转站文档；不清楚也没关系，直接填上 Key 就能用，插件会自动找出可用的参数格式。)

图片默认参数：

- `image.size`：默认像素尺寸。默认 `auto`。OpenAI 官方接口使用官方尺寸白名单；部分 GPT Image 2 兼容站支持更多 `宽x高`。
- `image.resolution`：默认清晰度档位，支持 `auto`、`1k`、`2k`、`4k`。OpenAI 官方接口会通过 `size` 映射表达，不额外发送 `resolution` 字段；兼容/逆向渠道支持时会生效。
- `image.aspect_ratio`：默认比例。支持比例参数的供应商会按比例出图；OpenAI 官方会近似映射为方图、横图或竖图。改图没传比例时优先按第一张参考图比例。
- `image.max_reference_images`：最多参考图数量，GPT 图像编辑接口最多 16 张。
- `runtime.retry_times`：网络抖动重试次数。默认 `1`，只用于下载图片等无副作用请求。生成/改图是 POST 任务提交，为避免超时后重复生成或重复扣费，插件不会自动重试这类请求。
- `prompt.prevent_prompt_rewrite`：默认开启，会在提示词前追加 `Use the following text as the complete prompt. Do not rewrite it:`，尽量避免 GPT 图像接口自行改写用户提示词。

`image.size`、`image.resolution` 和 `image.aspect_ratio` 是三个不同概念：`size` 是具体像素尺寸，`resolution` 是 1K/2K/4K 清晰度档位，`aspect_ratio` 是画面比例。命令里显式传 `--size`、`--ratio` 或 `--resolution` 时，显式参数优先。

`(ps:那些用官方标准 OpenAI 兼容接口的模型似乎不能用图片比例,只能用具体像素尺寸 但是网页逆向的可以使用图片比例,不能使用具体像素尺寸 官方标准 OpenAI 兼容接口的参数请参考官方文档https://platform.openai.com/docs/api-reference/images/create)`

## 命令

```text
/生图 一个二次元人物 
/生图 一只赛博猫坐在霓虹窗边 --ratio 16:9 --resolution 2k --quality high
/生图 扁平图标，一个透明背景的蓝色火箭 --transparent --style flat icon --ratio 1:1
/改图 把这张图任务二次元化，背景不变
/改图 把背景换成夜晚城市，保留人物姿势 --resolution 2k
/改图 把这几张图合成一张电影海报 --ratio 2:3 --quality high
```

可用参数：

- `--resolution` 或 `--res`：`1k`、`2k`、`4k`。
- `--ratio` 或 `--aspect-ratio`：`1:1`、`16:9`、`9:16`、`3:2`、`2:3`、`21:9` 等。
- `--size` 或 `-s`：像素尺寸，主要给 OpenAI 官方/兼容接口用，例如 `1024x1024`、`1536x1024`、`1024x1536`、`1824x1024`。
- `--quality` 或 `-q`：`auto`、`low`、`medium`、`high`。
- `--style`：额外风格词。
- `--transparent`：透明背景。

如果不添加参数则使用astrbot插件设置参数

命令会保留用户提示词原文；外层引号只用于分组，提交给上游时会去掉。自然语言里写“宽高比设为 16:9 / 比例 9:16 / 分辨率 2K”时，插件会自动抽取为接口参数，不需要额外写 `--ratio` 或 `--resolution`。

## Bot 自动调用

插件会把工具说明注入到 LLM 系统提示词里，教 bot：

- 文生图时根据用途选择合适比例，例如头像/图标 `1:1`，手机壁纸 `9:16`，电脑壁纸 `16:9`，横幅 `21:9`。
- 改图时使用 `use_reference_images=true`，除非用户明确要求新比例，否则让插件按原图比例处理。
- 分辨率只有用户明确要求 1K/2K/4K 时才传；否则使用 AstrBot 配置里的默认分辨率。

函数工具和工具使用说明在插件内默认开启。需要禁止 bot 自动调用时，不用在插件配置里找开关，直接到 AstrBot 自带的函数工具开关里关闭 `generate_image_with_gpt_image_2` 即可。


## 开发说明

- 配置只暴露一个 `api` 块。供应商类型由插件运行时自动判断：先按启发式或缓存选一套参数档案 (`standard` / `flexible`)，请求失败且像参数格式错误时再用另一套档案重试一次，成功后把档案按 `(base_url, model, operation)` 缓存到内存。
- 参数格式重试只在上游明确报参数格式错误(例如 `unknown parameter`、`size must be one of`、`resolution not supported` 这些)时触发；401/403/429、内容审核拒绝这类错误不会再浪费一次额度去重试。
- 参数档案切换只在 400/422 这类明确参数错误时触发。模糊 5xx 不再切档，避免上游其实已接收任务但返回异常时再次提交，造成一次请求生成多张图。
- 网络抖动重试由 `runtime.retry_times` 控制，只用于下载图片等无副作用请求。图片生成/编辑是 POST 任务提交，插件不会对这类请求做网络重试，避免上游实际已收到任务时重复生成或重复扣费。
- 所有出站请求默认带一个 `Mozilla/5.0 (compatible; AstrBotGPTImagePlugin/...)` 风格的 User-Agent，避开 Cloudflare 默认拦截 `Python/aiohttp` 这类特征 UA 的规则。可在 `api.user_agent` 覆盖。
- 上游 403/503 + 含 `cf-ray` / `cloudflare` 的 HTML 响应会被识别为 Cloudflare 拦截页，错误信息会带上 `cf-ray` ID，并提示用户改 `api.user_agent` 或联系中转放行 IP；不会把整页 HTML 直接糊给用户。
- 网络重试耗尽后区分超时和其他错误：超时会把 `api.timeout_seconds` 实际值带到错误信息里，方便用户知道该调哪个参数。
- 旧 `openai` / `two_api` 配置块仍然在代码里作为 fallback 读取，且尊重老的 `enabled` 开关。但只要用户在新 `api` 块里填了 Key，或把 `base_url`/`model`/`timeout_seconds` 改成非默认值，就视为已经迁移到新块，旧块里残留的 Key 不会再被偷偷打到新地址上。历史默认模型 `gpt-image-2` 不会单独触发“已迁移”判断，避免老用户升级后绕过旧配置。
- 多图编辑上传时重复使用 multipart 字段名 `image`，不要改成 `image[]`，否则严格兼容 OpenAI 的接口可能拒收。
- `size`、`resolution`、`aspect_ratio` 的归一化逻辑在 `main.py` 里集中处理。改动前建议分别用 OpenAI 标准接口和 2api 类接口验证两套档案路径。
- LLM 工具默认后台生成并保留任务引用，避免长时间生图完成后无人发送结果。
- 1.2.0 起 `openai` / `two_api` 两组配置在面板上合并为一个 `api` 块；老用户升级后无需手动迁移，等到方便时把旧块的内容拷过来再删掉旧块即可。
