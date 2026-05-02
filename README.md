# GPT Image 图像生成

GPT Image 图片生成插件，支持所有 GPT Image 系列模型。

支持手动使用 `/生图`、`/改图` 调用，也支持 LLM 通过函数工具自主调用。只要模型供应商提供(中转站也行) OpenAI 官方 Images API、标准 OpenAI 兼容接口，或网页(只有中转站) 2api/ToAPIs 接口，就可以通过配置模型名、Base URL 和 Key 接入。

- `/生图`：文生图。
- `/改图`：图生图/图片编辑，自动读取当前消息图片和回复/引用消息里的图片。
- 或者自然语言描述 模型会自动调用工具进行生成

`(ps:注意 OpenAI 官方 Images API、标准 OpenAI 兼容接口 只是一种格式,并不是代表你一定要用openai官方提供的api 只要你的中转站提供的是这个官方格式,那么你就可以用 OpenAI 官方 / 标准兼容接口 来生成图片)`

## 配置

在 AstrBot 插件配置里按模型供应商提供的接入方式选一个渠道填写即可。

## OpenAI 官方 / 标准兼容接口渠道：

-默认开启为此项 需要提供非网页逆向的渠道
-`OpenAI Base URL`填你模型供应商提供的网址就行,比如 https://astrbot.com/v1 默认选择模型名称 "gpt-image-2" 根据模型供应商提供的名称自行修改

## 网页逆向渠道(GPT给我插件那边写的是2api / ToAPIs 网页逆向渠道)：

-如果你的渠道是这种API,则需要关闭首项,使用此项
-`2api Base URL`填你模型供应商提供的网址就行,比如 https://astrbot.com/v1 默认选择模型名称 "gpt-image-2" 根据模型供应商提供的名称自行修改

(PS:网页逆向渠道只能生成1K分辨率模型,如果分不清模型是网页逆向还是兼容官方接口渠道的去看中转站模型介绍,基本上都会说这个模型是逆向还是可以通过官方API格式调用)

图片默认参数：

- `image.size`：默认像素尺寸。默认 `auto`，表示不主动传 `size`。OpenAI 官方通常只接受固定像素尺寸；部分供应商支持更多 `宽x高`。
- `image.resolution`：默认清晰度档位，支持 `auto`、`1k`、`2k`、`4k`。默认 `1k`，供应商支持时会生效，不支持时可能忽略。
- `image.aspect_ratio`：默认比例。支持比例参数的供应商会按比例出图；OpenAI 官方会近似映射为方图、横图或竖图。改图没传比例时优先按第一张参考图比例。
- `image.max_reference_images`：最多参考图数量，GPT 图像编辑接口最多 16 张。

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

## Bot 自动调用

插件会把工具说明注入到 LLM 系统提示词里，教 bot：

- 文生图时根据用途选择合适比例，例如头像/图标 `1:1`，手机壁纸 `9:16`，电脑壁纸 `16:9`，横幅 `21:9`。
- 改图时使用 `use_reference_images=true`，除非用户明确要求新比例，否则让插件按原图比例处理。
- 分辨率只有用户明确要求 1K/2K/4K 时才传；否则使用 AstrBot 配置里的默认分辨率。

函数工具和工具使用说明在插件内默认开启。需要禁止 bot 自动调用时，不用在插件配置里找开关，直接到 AstrBot 自带的函数工具开关里关闭 `generate_image_with_gpt_image_2` 即可。


## 开发说明

- 渠道选择只表示供应商提供的接口类型，不表示主备关系。`openai` 面向 OpenAI 官方/标准兼容接口，`two_api` 面向网页逆向/2api/ToAPIs 接口。
- 多图编辑上传时重复使用 multipart 字段名 `image`，不要改成 `image[]`，否则严格兼容 OpenAI 的接口可能拒收。
- `size`、`resolution`、`aspect_ratio` 的归一化逻辑在 `main.py` 里集中处理。改动前建议分别测试 OpenAI 官方接口和网页 2api 接口。
- LLM 工具默认后台生成并保留任务引用，避免长时间生图完成后无人发送结果。
