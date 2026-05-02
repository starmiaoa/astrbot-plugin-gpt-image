# GPT Image 图像生成

GPT Image 图片生成插件，支持所有 GPT Image 系列模型。

支持手动使用 `/生图`、`/改图` 调用，也支持 LLM 通过函数工具自主调用。只要模型供应商提供 OpenAI 官方 Images API、标准 OpenAI 兼容接口，或网页 2api/ToAPIs 接口，就可以通过配置模型名、Base URL 和 Key 接入。

- `/生图`：文生图。
- `/改图`：图生图/图片编辑，自动读取当前消息图片和回复/引用消息里的图片。
- `generate_image_with_gpt_image_2`：注入给 bot 的 LLM 函数工具，bot 可按用户需求自动调用。

## 配置

在 AstrBot 插件配置里按模型供应商提供的接入方式选一个渠道填写即可。

OpenAI 官方 / 标准兼容接口：

- `openai.enabled`：供应商提供 OpenAI 官方接口或标准 OpenAI-compatible Images API 时开启，默认开启。
- `openai.api_key`：OpenAI 或兼容站 API Key。留空时读取 `OPENAI_API_KEY`。
- `openai.base_url`：默认 `https://api.openai.com/v1`。填到 `/v1` 即可，插件会自动补 `/images/generations` 或 `/images/edits`。
- `openai.model`：默认 `gpt-image-2`，兼容站按实际模型名填写。

2api / ToAPIs 网页逆向渠道：

- `two_api.enabled`：供应商提供网页逆向/2api/ToAPIs 渠道时开启，并建议关闭 `openai.enabled`。
- `two_api.api_key`：网页逆向/2api/ToAPIs 渠道的 API Key。留空时读取 `TWO_API_KEY`，再回退到 `OPENAI_API_KEY`。
- `two_api.base_url`：2api/ToAPIs 的 `/v1` 地址或根地址。只填根地址时插件会自动补 `/v1/images/generations`。
- `two_api.model`：默认 `gpt-image-2`。不同网页逆向渠道可能需要 `gpt-image-2-all`、`gpt-image-1.5-official` 等模型名，按你的渠道要求填写。

图片默认参数：

- `image.size`：默认像素尺寸。默认 `auto`，表示不主动传 `size`。OpenAI 官方通常只接受固定像素尺寸；部分供应商支持更多 `宽x高`。
- `image.resolution`：默认清晰度档位，支持 `auto`、`1k`、`2k`、`4k`。默认 `1k`，供应商支持时会生效，不支持时可能忽略。
- `image.aspect_ratio`：默认比例。支持比例参数的供应商会按比例出图；OpenAI 官方会近似映射为方图、横图或竖图。改图没传比例时优先按第一张参考图比例。
- `image.max_reference_images`：最多参考图数量，GPT 图像编辑接口最多 16 张。

`image.size`、`image.resolution` 和 `image.aspect_ratio` 是三个不同概念：`size` 是具体像素尺寸，`resolution` 是 1K/2K/4K 清晰度档位，`aspect_ratio` 是画面比例。命令里显式传 `--size`、`--ratio` 或 `--resolution` 时，显式参数优先。

## 命令

```text
/生图 一只赛博猫坐在霓虹窗边 --ratio 16:9 --resolution 2k --quality high
/生图 扁平图标，一个透明背景的蓝色火箭 --transparent --style flat icon --ratio 1:1
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

## Bot 自动调用

插件会把工具说明注入到 LLM 系统提示词里，教 bot：

- 文生图时根据用途选择合适比例，例如头像/图标 `1:1`，手机壁纸 `9:16`，电脑壁纸 `16:9`，横幅 `21:9`。
- 改图时使用 `use_reference_images=true`，除非用户明确要求新比例，否则让插件按原图比例处理。
- 分辨率只有用户明确要求 1K/2K/4K 时才传；否则使用 AstrBot 配置里的默认分辨率。
- 工具返回后，bot 会按自己的人格设定用短句回复，例如“好，收到”“我开始画了”，完成后图片会自动发到当前聊天。

函数工具和工具使用说明在插件内默认开启。需要禁止 bot 自动调用时，不用在插件配置里找开关，直接到 AstrBot 自带的函数工具开关里关闭 `generate_image_with_gpt_image_2` 即可。

## 供应商渠道

OpenAI 官方 / 标准兼容接口和 2api / ToAPIs 网页逆向渠道走同一套 OpenAI-compatible Images API 调用逻辑，区别主要看模型供应商让你填哪一组接入信息：

1. 文生图调用 `/v1/images/generations`。
2. 改图调用 `/v1/images/edits`，多张参考图会重复使用 multipart 字段名 `image` 上传。
3. 返回 `b64_json` 时直接保存，返回 `url` 时下载图片并发送回当前聊天。

Base URL 填到 `/v1` 即可，例如 `https://api.example.com/v1`。如果只填根地址，插件会自动补 `/v1/images/generations` 或 `/v1/images/edits`。

## 开发说明

- 渠道选择只表示供应商提供的接口类型，不表示主备关系。`openai` 面向 OpenAI 官方/标准兼容接口，`two_api` 面向网页逆向/2api/ToAPIs 接口。
- 多图编辑上传时重复使用 multipart 字段名 `image`，不要改成 `image[]`，否则严格兼容 OpenAI 的接口可能拒收。
- `size`、`resolution`、`aspect_ratio` 的归一化逻辑在 `main.py` 里集中处理。改动前建议分别测试 OpenAI 官方接口和网页 2api 接口。
- LLM 工具默认后台生成并保留任务引用，避免长时间生图完成后无人发送结果。
