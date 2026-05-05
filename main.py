"""AstrBot plugin for GPT image generation models.

The provider selection in this file follows the way a model supplier exposes
its service: OpenAI official/compatible Images API, or webpage reverse
2api/ToAPIs endpoints that still accept OpenAI-compatible requests.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import re
import shlex
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp

from astrbot.api import llm_tool, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.provider.func_tool_manager import FunctionToolManager
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path


PLUGIN_ID = "astrbot_plugin_gpt_image_2"
TOOL_NAME = "generate_image_with_gpt_image_2"

MAX_GPT_IMAGE_REFERENCE_IMAGES = 16
MAX_REFERENCE_IMAGE_BYTES = 50 * 1024 * 1024

# GPT image providers do not all accept the same size syntax. The plugin keeps
# the public config broad, then normalizes the value according to the selected
# supplier before sending the request.
VALID_OPENAI_SIZES = {"auto", "1024x1024", "1536x1024", "1024x1536"}
VALID_ASPECT_RATIOS = {
    "auto",
    "1:1",
    "3:2",
    "2:3",
    "4:3",
    "3:4",
    "5:4",
    "4:5",
    "16:9",
    "9:16",
    "2:1",
    "1:2",
    "21:9",
    "9:21",
}
VALID_4K_ASPECT_RATIOS = {"16:9", "9:16", "2:1", "1:2", "21:9", "9:21"}
VALID_RESOLUTIONS = {"auto", "1k", "2k", "4k"}
VALID_QUALITIES = {"auto", "low", "medium", "high"}
VALID_OUTPUT_FORMATS = {"png", "jpeg", "webp"}
VALID_BACKGROUNDS = {"auto", "transparent", "opaque"}
VALID_INPUT_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}

OPENAI_SIZE_TO_RATIO = {
    "1024x1024": "1:1",
    "1536x1024": "3:2",
    "1024x1536": "2:3",
}

DEFAULT_TOOL_GUIDE = """
# GPT Image 画图工具使用说明

你可以调用 `generate_image_with_gpt_image_2` 工具生成或修改图片。这个工具会真的提交图片生成任务 普通文字问答不要调用它。

什么时候调用：
- 只有用户明确要求你现在输出一张图片时才调用 例如“画一张...”“生成一张图...”“出图...”“做一张海报/头像/壁纸/图标...”。
- 只有用户明确要求修改图片 且当前消息或引用消息里有图片时才设置 `use_reference_images=true` 例如“改这张图...”“把图里的...换成...”“参考上图生成...”。
- 如果用户想改图但没有发图或引用图 不要调用工具 先请用户发送或引用图片。
- 如果用户只是讨论图片 询问建议 写提示词 改提示词 翻译提示词 分析图片风格 规划设计方案 询问参数或问能不能生成 不要调用工具 直接用文字回答。
- 用户表达不明确时不要主动生成 先用文字确认是否需要现在出图。
- 同一条用户消息最多调用一次工具 先生成一张图 除非用户后续继续要求修改或再生成 不要在同一轮对话里重复调用

参数怎么传：
- `prompt` 写完整、具体、可独立理解的视觉需求。改图时说明要保留什么、改变什么、参考图之间的关系。
- `resolution` 是 GPT 图像模型的清晰度档位：用户明确要 1K/2K/4K 时传 `1k`/`2k`/`4k`；否则留空，插件会使用 AstrBot 插件配置里的默认分辨率。供应商如果不支持该字段，可能会忽略。
- `aspect_ratio` 是画面比例：文生图时请根据用途主动选择最合适比例，例如头像/图标 `1:1`，海报/手机壁纸 `9:16`，电脑壁纸/横版海报 `16:9`，产品横幅 `21:9`，普通横图 `3:2`，普通竖图 `2:3`；如果用户明确要求比例就按用户要求传。改图时除非用户明确要求新比例，否则留空，插件会尽量按第一张参考图比例。
- `size` 是精确像素尺寸，例如 `1024x1024`、`1536x1024`、`1024x1536`；部分供应商也支持 `1824x1024` 这类 `宽x高`。不要用 `size` 传 1K/2K/4K，清晰度档位请用 `resolution`；比例请用 `aspect_ratio`。
- `quality` 日常留空或 `auto`；用户要求精细、最终稿、高清时用 `high`。
- `transparent_background=true` 用于 logo、贴纸、图标、UI 素材、抠图、透明底。
- `style` 可放额外风格词，比如写实、动漫、水彩、产品渲染、扁平图标。
- GPT 图像编辑最多会读取 16 张参考图；插件会自动从当前消息和引用消息里收集。

工具返回状态后 只按你当前的人格和语气回复一句确认 让用户知道这次图像任务已经接下并开始处理 不要复述工具返回文本 不要补充技术细节或具体耗时 不要再次调用同一个工具
""".strip()


@register(
    PLUGIN_ID,
    "starmiaoa",
    "GPT Image 图片生成插件，支持所有 GPT Image 系列模型",
    "1.1.2",
)
class GPTImage2Plugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}
        self._data_dir = Path(get_astrbot_plugin_data_path()) / PLUGIN_ID
        self._image_dir = self._data_dir / "images"
        self._image_dir.mkdir(parents=True, exist_ok=True)

        max_concurrent = self._int_cfg("runtime", "max_concurrent", 1)
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent))
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._inflight_tool_tasks: dict[str, asyncio.Task[tuple[str, str | None]]] = {}
        self._inflight_tool_turns: dict[str, asyncio.Task[tuple[str, str | None]]] = {}

    @filter.command("生图")
    async def generate_command(self, event: AstrMessageEvent):
        event.stop_event()
        prompt = self._extract_command_prompt(event, ("生图",))
        opts, prompt = self._parse_inline_options(prompt)
        if not prompt:
            yield event.plain_result(
                "用法：/生图 一只赛博猫坐在霓虹窗边 --size 1024x1024 --quality high"
            )
            return

        task = asyncio.create_task(
            self._generate_image(
                prompt=prompt,
                size=opts.get("size"),
                aspect_ratio=opts.get("aspect_ratio"),
                resolution=opts.get("resolution"),
                quality=opts.get("quality"),
                style=opts.get("style"),
                transparent_background=opts.get("transparent_background", False),
            )
        )
        await asyncio.sleep(0)
        if not task.done():
            yield event.plain_result(self._tool_submitted_message())
        try:
            image_path, revised_prompt = await task
        except Exception as exc:
            logger.error("GPT Image /生图 command failed", exc_info=True)
            yield event.plain_result(f"图片生成失败：{self._friendly_error(exc)}")
            return

        yield event.chain_result(self._build_result_chain(image_path, revised_prompt))

    @filter.command("改图")
    async def edit_command(self, event: AstrMessageEvent):
        event.stop_event()
        prompt = self._extract_command_prompt(event, ("改图",))
        opts, prompt = self._parse_inline_options(prompt)
        if not prompt:
            yield event.plain_result(
                "用法：发送或引用图片后，输入 /改图 把背景换成夜晚城市 --size 1536x1024 --quality high"
            )
            return

        reference_paths = await self._collect_reference_image_paths(event)
        if not reference_paths:
            yield event.plain_result("没有找到可用参考图。请在同一条消息里发图，或回复/引用一张图后再使用 /改图。")
            return

        task = asyncio.create_task(
            self._generate_image(
                prompt=prompt,
                size=opts.get("size"),
                aspect_ratio=opts.get("aspect_ratio"),
                resolution=opts.get("resolution"),
                quality=opts.get("quality"),
                style=opts.get("style"),
                transparent_background=opts.get("transparent_background", False),
                reference_image_paths=reference_paths,
            )
        )
        await asyncio.sleep(0)
        if not task.done():
            yield event.plain_result(self._tool_submitted_message())
        try:
            image_path, revised_prompt = await task
        except Exception as exc:
            logger.error("GPT Image /改图 command failed", exc_info=True)
            yield event.plain_result(f"图片修改失败：{self._friendly_error(exc)}")
            return

        yield event.chain_result(self._build_result_chain(image_path, revised_prompt))

    @filter.command("生图帮助")
    async def gptimage_help(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(
            "GPT Image 图像生成插件已加载。\n"
            "命令：/生图 提示词；/改图 提示词（同消息发图，或回复/引用图片）。\n"
            "参数：--resolution 1k|2k|4k，--ratio 1:1|16:9|9:16|3:2|2:3，"
            "--size 1024x1024|1536x1024|1024x1536|auto，"
            "--quality auto|low|medium|high，--style 风格，--transparent。\n"
            "说明：像素尺寸用 --size，清晰度档位用 --resolution，比例用 --ratio。\n"
            f"多图参考：GPT 图像编辑接口最多支持 {MAX_GPT_IMAGE_REFERENCE_IMAGES} 张参考图。"
        )

    @llm_tool(name=TOOL_NAME)
    async def generate_image_with_gpt_image_2(
        self,
        event: AstrMessageEvent,
        prompt: str,
        size: str = "",
        aspect_ratio: str = "",
        resolution: str = "",
        quality: str = "",
        style: str = "",
        transparent_background: bool = False,
        use_reference_images: bool = False,
    ):
        """Generate or edit an image with a GPT image model and send it to the current chat.

        Use this tool only when the user explicitly asks you to generate or edit an
        image now. Set use_reference_images=true only when the user wants to edit,
        transform, combine, or follow images attached to the current message or quoted/replied
        message. Do not use it for normal text-only answers, prompt writing, design
        advice, image discussion, parameter questions, or ambiguous requests.

        Call this tool at most once for one user message. The tool sends the finished image directly to the user. After the tool reports
        background-task status, respond in your persona and naturally let the user know
        you accepted the image task and started working on it. Let length, phrasing, and
        punctuation follow your persona and context. Do not repeat the tool status text
        or mention technical details or timing.

        Args:
            prompt(string): Required. A complete, concrete image prompt describing subject, style, composition, colors, lighting, text to include if any, and constraints from the user.
            size(string): Optional. Pixel size for OpenAI official/compatible APIs. Only use when the user explicitly gives a pixel size, such as 1024x1024, 1536x1024, 1024x1536, 1824x1024, or auto. Leave empty otherwise.
            aspect_ratio(string): Optional. Desired aspect ratio, such as 1:1, 16:9, 9:16, 3:2, 2:3, 21:9. For text-to-image, choose the best ratio for the user's intended use when they do not specify one. For image editing, leave empty unless the user asks to change the ratio; the plugin will prefer the first reference image's ratio.
            resolution(string): Optional. GPT image resolution tier: auto, 1k, 2k, or 4k. Use it only when the user explicitly asks for a resolution; otherwise leave empty so the AstrBot plugin setting is used. Some providers may ignore unsupported tiers.
            quality(string): Optional. Image quality: auto, low, medium, or high. Use high only when the user asks for detailed/final artwork.
            style(string): Optional. Extra visual style words such as realistic, anime, watercolor, product render, poster, pixel art, or flat icon.
            transparent_background(boolean): Optional. Set true for logos, stickers, icons, UI assets, transparent cutouts, or when the user asks for no background.
            use_reference_images(boolean): Optional. Set true for image editing/image-to-image requests using images in the current or quoted message. Up to 16 images are used.

        """
        if not prompt or not prompt.strip():
            return "缺少图片提示词，无法生成或修改图片。"

        reference_paths: list[str] = []
        if use_reference_images:
            reference_paths = await self._collect_reference_image_paths(event)
            if not reference_paths:
                return "没有找到参考图。请先发送或引用一张图片，再说明要怎么改。"

        turn_key = self._tool_turn_key(event)
        turn_task = self._inflight_tool_turns.get(turn_key)
        if turn_task and not turn_task.done():
            return self._tool_submitted_message(reference_count=len(reference_paths))
        if turn_task and turn_task.done():
            self._inflight_tool_turns.pop(turn_key, None)

        request_key = self._tool_request_key(
            event,
            prompt=prompt,
            size=size,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            quality=quality,
            style=style,
            transparent_background=transparent_background,
            reference_paths=reference_paths,
        )
        existing_task = self._inflight_tool_tasks.get(request_key)
        if existing_task and not existing_task.done():
            return self._tool_submitted_message(reference_count=len(reference_paths))
        if existing_task and existing_task.done():
            self._inflight_tool_tasks.pop(request_key, None)

        wait_seconds = max(0, self._int_cfg("runtime", "llm_tool_foreground_wait_seconds", 0))
        task = asyncio.create_task(
            self._generate_image(
                prompt=prompt,
                size=size,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                quality=quality,
                style=style,
                transparent_background=transparent_background,
                reference_image_paths=reference_paths,
            )
        )
        self._inflight_tool_tasks[request_key] = task
        self._inflight_tool_turns[turn_key] = task

        def _clear_inflight(done_task: asyncio.Task[tuple[str, str | None]]) -> None:
            if self._inflight_tool_tasks.get(request_key) is done_task:
                self._inflight_tool_tasks.pop(request_key, None)
            if self._inflight_tool_turns.get(turn_key) is done_task:
                self._inflight_tool_turns.pop(turn_key, None)

        task.add_done_callback(_clear_inflight)

        try:
            if wait_seconds == 0:
                raise asyncio.TimeoutError
            image_path, revised_prompt = await asyncio.wait_for(
                asyncio.shield(task),
                timeout=wait_seconds,
            )
        except asyncio.TimeoutError:
            self._schedule_background_delivery(event, task)
            return self._tool_submitted_message(reference_count=len(reference_paths))
        except Exception as exc:
            logger.error("GPT Image tool failed", exc_info=True)
            return f"图片生成失败：{self._friendly_error(exc)}"

        return event.chain_result(self._build_result_chain(image_path, revised_prompt))

    @filter.on_llm_request(priority=-9000)
    async def teach_llm_when_to_use_tool(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        guide = self._str_cfg("runtime", "tool_instruction", "") or DEFAULT_TOOL_GUIDE
        if guide.strip():
            req.system_prompt += f"\n\n{guide.strip()}\n"

    async def _generate_image(
        self,
        *,
        prompt: str,
        size: str | None = None,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        quality: str | None = None,
        style: str | None = None,
        transparent_background: bool = False,
        reference_image_paths: list[str] | None = None,
    ) -> tuple[str, str | None]:
        self._ensure_enabled()
        provider_mode = self._provider_mode()
        api_key = self._api_key(provider_mode)
        if not api_key:
            provider_name = "2api/ToAPIs" if provider_mode == "2api" else "OpenAI 官方"
            raise RuntimeError(f"{provider_name} 配置里没有填写 api_key，也没有设置可用的环境变量。")

        references = self._prepare_reference_image_paths(reference_image_paths or [])
        final_prompt = self._build_prompt(prompt, style)
        payload = self._build_payload(
            final_prompt,
            provider_mode=provider_mode,
            size=size,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            quality=quality,
            transparent_background=transparent_background,
            reference_image_paths=references,
        )

        async with self._semaphore:
            if references:
                response = await self._post_images_edit_api(api_key, payload, references, provider_mode=provider_mode)
            else:
                response = await self._post_images_generation_api(api_key, payload, provider_mode=provider_mode)
            image_path, revised_prompt = await self._save_image_from_response(response)
            self._cleanup_old_images()
            return image_path, revised_prompt

    async def _post_images_generation_api(
        self,
        api_key: str,
        payload: dict[str, Any],
        *,
        provider_mode: str,
    ) -> dict[str, Any]:
        url = self._images_generation_url(provider_mode)
        headers = self._api_headers(api_key, content_type="application/json")

        timeout = aiohttp.ClientTimeout(total=max(10, self._timeout_seconds(provider_mode)))
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(self._format_api_error(resp.status, text))
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("图像接口返回了无法解析的 JSON。") from exc

    async def _post_images_edit_api(
        self,
        api_key: str,
        payload: dict[str, Any],
        image_paths: list[str],
        *,
        provider_mode: str,
    ) -> dict[str, Any]:
        url = self._images_edit_url(provider_mode)
        headers = self._api_headers(api_key)
        form = aiohttp.FormData()

        for key, value in payload.items():
            form.add_field(key, str(value))

        opened_files = []
        try:
            for image_path in image_paths:
                content_type = self._detect_image_content_type(image_path)
                file_obj = open(image_path, "rb")
                opened_files.append(file_obj)
                form.add_field(
                    "image",
                    file_obj,
                    filename=self._multipart_filename(image_path, content_type),
                    content_type=content_type,
                )

            timeout = aiohttp.ClientTimeout(total=max(10, self._timeout_seconds(provider_mode)))
            async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                async with session.post(url, headers=headers, data=form) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(self._format_api_error(resp.status, text))
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("图像编辑接口返回了无法解析的 JSON。") from exc
        finally:
            for file_obj in opened_files:
                try:
                    file_obj.close()
                except Exception:
                    pass

    async def _save_image_from_response(self, response: dict[str, Any]) -> tuple[str, str | None]:
        data = response.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError("图像接口没有返回 data[0]。")

        item = data[0]
        if not isinstance(item, dict):
            raise RuntimeError("图像接口返回格式异常：data[0] 不是对象。")

        revised_prompt = item.get("revised_prompt")
        output_format = self._output_format()
        suffix = ".jpg" if output_format == "jpeg" else f".{output_format}"
        file_path = self._image_dir / f"gpt_image_2_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"

        b64_json = item.get("b64_json")
        if isinstance(b64_json, str) and b64_json.strip():
            raw_b64 = b64_json.split(",", 1)[-1] if "," in b64_json[:64] else b64_json
            image_bytes = base64.b64decode(raw_b64)
            file_path.write_bytes(image_bytes)
            return str(file_path), revised_prompt

        image_url = item.get("url")
        if isinstance(image_url, str) and image_url.startswith(("http://", "https://")):
            await self._download_image(image_url, file_path)
            return str(file_path), revised_prompt

        raise RuntimeError("图像接口没有返回 b64_json 或 url。")

    async def _download_image(self, url: str, file_path: Path) -> None:
        timeout = aiohttp.ClientTimeout(total=max(10, self._timeout_seconds(self._provider_mode())))
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.get(url) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f"下载图片失败：HTTP {resp.status} {text[:300]}")
                file_path.write_bytes(await resp.read())

    def _schedule_background_delivery(
        self,
        event: AstrMessageEvent,
        task: asyncio.Task[tuple[str, str | None]],
    ) -> None:
        # Keep a strong reference to the background sender. Long image jobs can
        # outlive the tool call, and losing this task would mean the image is
        # generated but never sent back to the chat.
        async def _deliver():
            try:
                image_path, revised_prompt = await task
                await event.send(event.chain_result(self._build_result_chain(image_path, revised_prompt)))
            except Exception as exc:
                logger.error("GPT Image background delivery failed", exc_info=True)
                try:
                    await event.send(event.plain_result(f"图片生成失败：{self._friendly_error(exc)}"))
                except Exception:
                    logger.error("Failed to send GPT Image background error", exc_info=True)

        sender = asyncio.create_task(_deliver())
        self._background_tasks.add(sender)

        def _report_done(done_task: asyncio.Task):
            self._background_tasks.discard(done_task)
            if done_task.cancelled():
                return
            exc = done_task.exception()
            if exc:
                logger.error(
                    "GPT Image background sender crashed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

        sender.add_done_callback(_report_done)

    def _build_payload(
        self,
        prompt: str,
        *,
        provider_mode: str,
        size: str | None,
        aspect_ratio: str | None,
        resolution: str | None,
        quality: str | None,
        transparent_background: bool,
        reference_image_paths: list[str],
    ) -> dict[str, Any]:
        model = self._model_name(provider_mode)
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": 1,
        }

        # Standard OpenAI endpoints prefer concrete pixel sizes. Webpage
        # reverse/2api suppliers often accept aspect ratios through the size
        # field, so this branch intentionally preserves ratios for that channel.
        if provider_mode == "2api":
            normalized_size = self._normalize_compatible_size(
                size,
                aspect_ratio=aspect_ratio,
                reference_image_paths=reference_image_paths,
                prefer_reference_ratio=bool(reference_image_paths),
            )
        else:
            normalized_size = self._normalize_openai_size(
                size,
                aspect_ratio=aspect_ratio,
                reference_image_paths=reference_image_paths,
                prefer_reference_ratio=bool(reference_image_paths),
            )
        if normalized_size != "auto":
            payload["size"] = normalized_size

        normalized_ratio = self._normalize_aspect_ratio(
            aspect_ratio,
            size=size,
            reference_image_paths=reference_image_paths,
            prefer_reference_ratio=bool(reference_image_paths),
        )
        normalized_resolution = self._normalize_resolution(resolution)
        if normalized_resolution != "auto":
            if provider_mode == "2api" and normalized_resolution == "4k" and normalized_ratio not in VALID_4K_ASPECT_RATIOS:
                normalized_resolution = "2k"
            payload["resolution"] = normalized_resolution

        normalized_quality = self._normalize_quality(quality)
        if normalized_quality != "auto":
            payload["quality"] = normalized_quality

        configured_background = self._str_cfg("image", "background", "auto")
        output_format = self._output_format()
        if output_format == "jpeg" and (transparent_background or configured_background == "transparent"):
            output_format = "png"
        payload["output_format"] = output_format

        compression = self._int_cfg("image", "output_compression", 0)
        if output_format in {"jpeg", "webp"} and 0 < compression <= 100:
            payload["output_compression"] = compression

        background = "transparent" if transparent_background else configured_background
        background = background if background in VALID_BACKGROUNDS else "auto"
        if background != "auto":
            payload["background"] = background

        moderation = self._str_cfg("image", "moderation", "")
        if moderation:
            payload["moderation"] = moderation

        return payload

    def _build_prompt(self, prompt: str, style: str | None) -> str:
        parts = []
        prefix = self._str_cfg("prompt", "prompt_prefix", "")
        suffix = self._str_cfg("prompt", "prompt_suffix", "")
        negative = self._str_cfg("prompt", "negative_prompt", "")

        if prefix:
            parts.append(prefix.strip())
        parts.append(prompt.strip())
        if style and style.strip() and style.strip().lower() != "auto":
            parts.append(f"Visual style: {style.strip()}.")
        if suffix:
            parts.append(suffix.strip())
        if negative:
            parts.append(f"Avoid: {negative.strip()}.")

        return "\n\n".join(parts).strip()

    def _build_result_chain(self, image_path: str, revised_prompt: str | None) -> list[Any]:
        del revised_prompt
        return [Image.fromFileSystem(image_path)]

    def _tool_submitted_message(self, *, reference_count: int = 0) -> str:
        del reference_count
        return "收到 开始画了"

    def _tool_request_key(
        self,
        event: AstrMessageEvent,
        *,
        prompt: str,
        size: str,
        aspect_ratio: str,
        resolution: str,
        quality: str,
        style: str,
        transparent_background: bool,
        reference_paths: list[str],
    ) -> str:
        payload = {
            "scope": self._event_scope_key(event),
            "prompt": prompt.strip(),
            "size": (size or "").strip(),
            "aspect_ratio": (aspect_ratio or "").strip(),
            "resolution": (resolution or "").strip(),
            "quality": (quality or "").strip(),
            "style": (style or "").strip(),
            "transparent_background": bool(transparent_background),
            "reference_paths": [str(Path(path).resolve()) for path in reference_paths],
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _tool_turn_key(self, event: AstrMessageEvent) -> str:
        payload = {
            "scope": self._event_scope_key(event),
            "message": self._event_message_key(event),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _event_message_key(self, event: AstrMessageEvent) -> str:
        parts: list[str] = []
        message_obj = getattr(event, "message_obj", None)
        for name in ("message_id", "msg_id", "id", "seq"):
            value = getattr(message_obj, name, None)
            if value:
                parts.append(str(value))

        for name in ("get_message_id", "get_msg_id"):
            method = getattr(event, name, None)
            if not callable(method):
                continue
            try:
                value = method()
            except Exception:
                continue
            if value:
                parts.append(str(value))

        text = ""
        try:
            text = str(getattr(event, "message_str", "") or event.get_message_str() or "")
        except Exception:
            text = str(getattr(event, "message_str", "") or "")
        if text:
            parts.append(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])

        return "|".join(dict.fromkeys(parts)) or "unknown-message"

    def _event_scope_key(self, event: AstrMessageEvent) -> str:
        parts: list[str] = []
        for name in ("unified_msg_origin", "session_id"):
            value = getattr(event, name, None)
            if value:
                parts.append(str(value))

        message_obj = getattr(event, "message_obj", None)
        for name in ("session_id", "group_id", "sender_id", "user_id"):
            value = getattr(message_obj, name, None)
            if value:
                parts.append(str(value))

        for name in ("get_session_id", "get_group_id", "get_sender_id"):
            method = getattr(event, name, None)
            if not callable(method):
                continue
            try:
                value = method()
            except Exception:
                continue
            if value:
                parts.append(str(value))

        return "|".join(dict.fromkeys(parts)) or "global"

    def _parse_inline_options(self, text: str) -> tuple[dict[str, Any], str]:
        opts: dict[str, Any] = {
            "size": "",
            "aspect_ratio": "",
            "resolution": "",
            "quality": "",
            "style": "",
            "transparent_background": False,
        }
        if not text:
            return opts, ""

        try:
            tokens = shlex.split(text)
        except ValueError:
            tokens = text.split()

        prompt_parts: list[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in {"--size", "-s"} and i + 1 < len(tokens):
                self._assign_size_like_option(opts, tokens[i + 1])
                i += 2
                continue
            if token.startswith("--size="):
                self._assign_size_like_option(opts, token.split("=", 1)[1])
                i += 1
                continue
            if token in {"--ratio", "--aspect", "--aspect-ratio", "--比例"} and i + 1 < len(tokens):
                opts["aspect_ratio"] = tokens[i + 1]
                i += 2
                continue
            if token.startswith(("--ratio=", "--aspect=", "--aspect-ratio=", "--比例=")):
                opts["aspect_ratio"] = token.split("=", 1)[1]
                i += 1
                continue
            if token in {"--resolution", "--res", "--分辨率"} and i + 1 < len(tokens):
                opts["resolution"] = tokens[i + 1]
                i += 2
                continue
            if token.startswith(("--resolution=", "--res=", "--分辨率=")):
                opts["resolution"] = token.split("=", 1)[1]
                i += 1
                continue
            if token in {"--quality", "-q"} and i + 1 < len(tokens):
                opts["quality"] = tokens[i + 1]
                i += 2
                continue
            if token.startswith("--quality="):
                opts["quality"] = token.split("=", 1)[1]
                i += 1
                continue
            if token == "--style" and i + 1 < len(tokens):
                opts["style"] = tokens[i + 1]
                i += 2
                continue
            if token.startswith("--style="):
                opts["style"] = token.split("=", 1)[1]
                i += 1
                continue
            if token in {"--transparent", "--transparent-background", "--no-bg"}:
                opts["transparent_background"] = True
                i += 1
                continue
            prompt_parts.append(token)
            i += 1

        return opts, " ".join(prompt_parts).strip()

    def _assign_size_like_option(self, opts: dict[str, Any], value: str) -> None:
        normalized = (value or "").strip().lower()
        if re.fullmatch(r"\d+[x:：]\d+", normalized):
            if ":" in normalized or "：" in normalized:
                opts["aspect_ratio"] = normalized.replace("：", ":")
            else:
                opts["size"] = normalized
            return
        if normalized in VALID_RESOLUTIONS:
            opts["resolution"] = normalized
            return
        opts["size"] = value

    def _extract_command_prompt(self, event: AstrMessageEvent, command_names: tuple[str, ...]) -> str:
        text = str(event.message_str or event.get_message_str()).strip()
        for name in command_names:
            for prefix in (f"/{name}", name):
                if text == prefix:
                    return ""
                if text.startswith(prefix + " "):
                    return text[len(prefix) :].strip()
        return text

    async def _collect_reference_image_paths(self, event: AstrMessageEvent) -> list[str]:
        max_images = self._max_reference_images()
        paths: list[str] = []
        seen: set[str] = set()

        async def add_image(component: Image) -> None:
            if len(paths) >= max_images:
                return
            try:
                image_path = await component.convert_to_file_path()
            except Exception:
                logger.warning("Failed to convert AstrBot image component to file path", exc_info=True)
                return
            normalized = os.path.abspath(image_path)
            if normalized not in seen:
                seen.add(normalized)
                paths.append(normalized)

        async def walk(chain: Any) -> None:
            if not chain or len(paths) >= max_images:
                return
            if isinstance(chain, Image):
                await add_image(chain)
                return
            if not isinstance(chain, (list, tuple)):
                return

            for component in chain:
                if len(paths) >= max_images:
                    break
                if isinstance(component, Image):
                    await add_image(component)

                nested_chain = getattr(component, "chain", None)
                if nested_chain:
                    await walk(nested_chain)

                nested_content = getattr(component, "content", None)
                if nested_content:
                    await walk(nested_content)

                nested_nodes = getattr(component, "nodes", None)
                if nested_nodes:
                    for node in nested_nodes:
                        await walk(getattr(node, "content", None))

        try:
            message_chain = event.get_messages()
        except Exception:
            message_chain = getattr(getattr(event, "message_obj", None), "message", [])

        await walk(message_chain)
        return paths

    def _prepare_reference_image_paths(self, paths: list[str]) -> list[str]:
        prepared: list[str] = []
        for image_path in paths[: self._max_reference_images()]:
            path = os.path.abspath(image_path)
            if not os.path.exists(path):
                raise RuntimeError(f"参考图不存在：{path}")
            size = os.path.getsize(path)
            if size > MAX_REFERENCE_IMAGE_BYTES:
                raise RuntimeError(f"参考图超过 50MB：{Path(path).name}")
            content_type = self._detect_image_content_type(path)
            if content_type not in VALID_INPUT_CONTENT_TYPES:
                raise RuntimeError(f"参考图格式不支持：{Path(path).name}。请使用 png、jpg/jpeg 或 webp。")
            prepared.append(path)
        return prepared

    def _max_reference_images(self) -> int:
        configured = self._int_cfg("image", "max_reference_images", MAX_GPT_IMAGE_REFERENCE_IMAGES)
        return min(MAX_GPT_IMAGE_REFERENCE_IMAGES, max(1, configured))

    def _detect_image_content_type(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as file_obj:
                header = file_obj.read(16)
        except Exception:
            header = b""

        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if header.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return "image/webp"

        guessed, _ = mimetypes.guess_type(image_path)
        return guessed or "application/octet-stream"

    def _multipart_filename(self, image_path: str, content_type: str) -> str:
        path = Path(image_path)
        filename = path.name or f"reference_{uuid.uuid4().hex[:8]}"
        if path.suffix:
            return filename

        extension = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
        }.get(content_type, "")
        return f"{filename}{extension}"

    def _normalize_openai_size(
        self,
        size: str | None,
        *,
        aspect_ratio: str | None,
        reference_image_paths: list[str],
        prefer_reference_ratio: bool,
    ) -> str:
        explicit_size = self._normalize_pixel_size(size)
        if explicit_size:
            return explicit_size

        ratio = self._normalize_aspect_ratio(
            aspect_ratio,
            size=None,
            reference_image_paths=reference_image_paths,
            prefer_reference_ratio=prefer_reference_ratio,
        )
        if ratio != "auto":
            return self._openai_size_for_ratio(ratio)

        configured_size = self._normalize_pixel_size(self._str_cfg("image", "size", "1024x1024"))
        if configured_size:
            return configured_size
        return "1024x1024"

    def _normalize_pixel_size(self, size: str | None) -> str:
        value = (size or "").strip().lower()
        if not value:
            return ""
        value = value.replace("*", "x").replace(" ", "")
        if value in VALID_OPENAI_SIZES:
            return value
        if re.fullmatch(r"\d{3,4}x\d{3,4}", value):
            return value
        return ""

    def _normalize_aspect_ratio(
        self,
        aspect_ratio: str | None,
        *,
        size: str | None,
        reference_image_paths: list[str],
        prefer_reference_ratio: bool,
    ) -> str:
        # Priority order is explicit request, explicit size, reference image,
        # plugin defaults, then auto. This keeps image editing close to the
        # original picture unless the user or bot asks for a new ratio.
        explicit_ratio = self._parse_aspect_ratio(aspect_ratio)
        if explicit_ratio:
            return explicit_ratio

        size_ratio = self._parse_aspect_ratio(size)
        if size_ratio:
            return size_ratio

        size_value = self._normalize_pixel_size(size)
        if size_value in OPENAI_SIZE_TO_RATIO:
            return OPENAI_SIZE_TO_RATIO[size_value]

        if prefer_reference_ratio and reference_image_paths:
            reference_ratio = self._reference_image_aspect_ratio(reference_image_paths[0])
            if reference_ratio:
                return reference_ratio

        configured_ratio = self._parse_aspect_ratio(self._str_cfg("image", "aspect_ratio", "auto"))
        if configured_ratio:
            return configured_ratio

        configured_size = self._str_cfg("image", "size", "")
        configured_size_ratio = self._parse_aspect_ratio(configured_size)
        if configured_size_ratio:
            return configured_size_ratio
        configured_pixel_size = self._normalize_pixel_size(configured_size)
        if configured_pixel_size in OPENAI_SIZE_TO_RATIO:
            return OPENAI_SIZE_TO_RATIO[configured_pixel_size]

        return "auto"

    def _parse_aspect_ratio(self, value: str | None) -> str:
        text = (value or "").strip().lower().replace("：", ":")
        if not text:
            return ""
        if text == "auto":
            return "auto"
        match = re.fullmatch(r"(\d{1,2})\s*:\s*(\d{1,2})", text)
        if not match:
            return ""
        left = int(match.group(1))
        right = int(match.group(2))
        if left <= 0 or right <= 0:
            return ""
        ratio = f"{left}:{right}"
        if ratio in VALID_ASPECT_RATIOS:
            return ratio
        return self._nearest_supported_aspect_ratio(left / right)

    def _reference_image_aspect_ratio(self, image_path: str) -> str:
        size = self._image_dimensions(image_path)
        if not size:
            return "auto"
        width, height = size
        if width <= 0 or height <= 0:
            return "auto"
        return self._nearest_supported_aspect_ratio(width / height)

    def _nearest_supported_aspect_ratio(self, ratio: float) -> str:
        candidates = [item for item in VALID_ASPECT_RATIOS if item != "auto"]

        def ratio_value(item: str) -> float:
            left, right = item.split(":", 1)
            return int(left) / int(right)

        return min(candidates, key=lambda item: abs(ratio_value(item) - ratio))

    def _openai_size_for_ratio(self, ratio: str) -> str:
        if ratio == "1:1":
            return "1024x1024"
        left, right = ratio.split(":", 1)
        ratio_value = int(left) / int(right)
        if ratio_value > 1:
            return "1536x1024"
        return "1024x1536"

    def _normalize_compatible_size(
        self,
        size: str | None,
        *,
        aspect_ratio: str | None,
        reference_image_paths: list[str],
        prefer_reference_ratio: bool,
    ) -> str:
        explicit_size = self._normalize_pixel_size(size)
        if explicit_size:
            return explicit_size

        explicit_ratio = self._parse_aspect_ratio(size)
        if explicit_ratio:
            return explicit_ratio

        ratio = self._normalize_aspect_ratio(
            aspect_ratio,
            size=None,
            reference_image_paths=reference_image_paths,
            prefer_reference_ratio=prefer_reference_ratio,
        )
        if ratio != "auto":
            return ratio

        configured_size = self._str_cfg("image", "size", "")
        configured_pixel_size = self._normalize_pixel_size(configured_size)
        if configured_pixel_size:
            return configured_pixel_size
        configured_ratio = self._parse_aspect_ratio(configured_size)
        if configured_ratio:
            return configured_ratio
        return "auto"

    def _normalize_resolution(self, resolution: str | None) -> str:
        value = (resolution or "").strip().lower()
        if not value:
            value = self._str_cfg("image", "resolution", "auto").lower()
        value = value.replace(" ", "")
        if value in {"1", "1k", "1024", "1024p"}:
            return "1k"
        if value in {"2", "2k", "2048", "2048p"}:
            return "2k"
        if value in {"4", "4k", "4096", "4096p"}:
            return "4k"
        return value if value in VALID_RESOLUTIONS else "auto"

    def _image_dimensions(self, image_path: str) -> tuple[int, int] | None:
        try:
            from PIL import Image as PILImage

            with PILImage.open(image_path) as image:
                return image.size
        except Exception:
            pass

        try:
            with open(image_path, "rb") as file_obj:
                data = file_obj.read(256 * 1024)
        except Exception:
            return None

        if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
            return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")

        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return self._webp_dimensions(data)

        if data.startswith(b"\xff\xd8"):
            return self._jpeg_dimensions(data)

        return None

    def _jpeg_dimensions(self, data: bytes) -> tuple[int, int] | None:
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            i += 2
            if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
                continue
            if i + 2 > len(data):
                return None
            segment_length = int.from_bytes(data[i : i + 2], "big")
            if segment_length < 2 or i + segment_length > len(data):
                return None
            if marker in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                height = int.from_bytes(data[i + 3 : i + 5], "big")
                width = int.from_bytes(data[i + 5 : i + 7], "big")
                return width, height
            i += segment_length
        return None

    def _webp_dimensions(self, data: bytes) -> tuple[int, int] | None:
        chunk_type = data[12:16]
        if chunk_type == b"VP8X" and len(data) >= 30:
            width = int.from_bytes(data[24:27], "little") + 1
            height = int.from_bytes(data[27:30], "little") + 1
            return width, height
        if chunk_type == b"VP8 " and len(data) >= 30:
            width = int.from_bytes(data[26:28], "little") & 0x3FFF
            height = int.from_bytes(data[28:30], "little") & 0x3FFF
            return width, height
        if chunk_type == b"VP8L" and len(data) >= 25:
            b0, b1, b2, b3 = data[21], data[22], data[23], data[24]
            width = 1 + (((b1 & 0x3F) << 8) | b0)
            height = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
            return width, height
        return None

    def _normalize_quality(self, quality: str | None) -> str:
        value = (quality or "").strip().lower()
        if not value:
            value = self._str_cfg("image", "quality", "auto").lower()
        return value if value in VALID_QUALITIES else "auto"

    def _output_format(self) -> str:
        value = self._str_cfg("image", "output_format", "png").lower()
        return value if value in VALID_OUTPUT_FORMATS else "png"

    def _images_generation_url(self, provider_mode: str | None = None) -> str:
        provider_mode = provider_mode or self._provider_mode()
        base_url = self._base_url(provider_mode).rstrip("/")
        if base_url.endswith("/images/generations"):
            return base_url
        if base_url.endswith("/images/edits"):
            return f"{base_url.rsplit('/images/', 1)[0]}/images/generations"
        if base_url.endswith("/v1"):
            return f"{base_url}/images/generations"
        return f"{base_url}/v1/images/generations"

    def _images_edit_url(self, provider_mode: str | None = None) -> str:
        provider_mode = provider_mode or self._provider_mode()
        base_url = self._base_url(provider_mode).rstrip("/")
        if base_url.endswith("/images/edits"):
            return base_url
        if base_url.endswith("/images/generations"):
            return f"{base_url.rsplit('/images/', 1)[0]}/images/edits"
        if base_url.endswith("/v1"):
            return f"{base_url}/images/edits"
        return f"{base_url}/v1/images/edits"

    def _provider_mode(self) -> str:
        # The two config groups are not primary/fallback. They represent the
        # interface type provided by the model supplier.
        openai_enabled = self._provider_enabled("openai")
        two_api_enabled = self._provider_enabled("2api")
        if two_api_enabled and not openai_enabled:
            return "2api"
        if openai_enabled:
            return "openai"
        if two_api_enabled:
            return "2api"

        legacy_mode = self._str_cfg("api", "provider_mode", "auto").lower()
        if legacy_mode in {"2api", "toapis", "toapi"}:
            return "2api"
        return "openai"

    def _provider_enabled(self, provider_mode: str) -> bool:
        if provider_mode == "2api":
            section = self._section("two_api")
            if "enabled" in section:
                return self._bool_cfg("two_api", "enabled", False)
            return self._str_cfg("api", "provider_mode", "").lower() in {"2api", "toapis", "toapi"}

        section = self._section("openai")
        if "enabled" in section:
            return self._bool_cfg("openai", "enabled", True)
        legacy_mode = self._str_cfg("api", "provider_mode", "openai").lower()
        return legacy_mode not in {"2api", "toapis", "toapi"}

    def _model_name(self, provider_mode: str) -> str:
        if provider_mode == "2api":
            model_2api = self._str_cfg("two_api", "model", "") or self._str_cfg("api", "model_2api", "")
            if model_2api:
                return model_2api
            configured_model = self._str_cfg("api", "model", "")
            if configured_model and configured_model != "gpt-image-2":
                return configured_model
            return "gpt-image-2"
        return self._str_cfg("openai", "model", "") or self._str_cfg("api", "model", "gpt-image-2") or "gpt-image-2"

    def _base_url(self, provider_mode: str) -> str:
        if provider_mode == "2api":
            return self._str_cfg("two_api", "base_url", "") or self._str_cfg("api", "base_url", "https://api.openai.com/v1") or "https://api.openai.com/v1"
        return self._str_cfg("openai", "base_url", "") or self._str_cfg("api", "base_url", "https://api.openai.com/v1") or "https://api.openai.com/v1"

    def _timeout_seconds(self, provider_mode: str) -> int:
        if provider_mode == "2api":
            return self._int_cfg("two_api", "timeout_seconds", self._int_cfg("api", "timeout_seconds", 180))
        return self._int_cfg("openai", "timeout_seconds", self._int_cfg("api", "timeout_seconds", 180))

    def _extract_image_response(self, data: Any) -> dict[str, Any] | None:
        if isinstance(data, dict):
            payload_data = data.get("data")
            if isinstance(payload_data, list) and payload_data:
                if any(isinstance(item, dict) and (item.get("b64_json") or item.get("url")) for item in payload_data):
                    return {"data": payload_data}

            nested = data.get("result") or data.get("output")
            if nested is not None:
                nested_response = self._extract_image_response(nested)
                if nested_response:
                    return nested_response

            for key in ("images", "urls", "image_urls", "result_images"):
                nested_images = data.get(key)
                if nested_images is not None:
                    nested_response = self._extract_image_response(nested_images)
                    if nested_response:
                        return nested_response

            if isinstance(payload_data, dict):
                nested_response = self._extract_image_response(payload_data)
                if nested_response:
                    return nested_response

            url = data.get("url") or data.get("image_url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return {"data": [{"url": url}]}
            b64_json = data.get("b64_json") or data.get("base64")
            if isinstance(b64_json, str) and b64_json.strip():
                return {"data": [{"b64_json": b64_json}]}

        if isinstance(data, list):
            image_items = []
            for item in data:
                if isinstance(item, str) and item.startswith(("http://", "https://")):
                    image_items.append({"url": item})
                elif isinstance(item, dict) and (item.get("url") or item.get("b64_json") or item.get("image_url")):
                    image_items.append(
                        {
                            "url": item.get("url") or item.get("image_url"),
                            "b64_json": item.get("b64_json"),
                        }
                    )
            if image_items:
                return {"data": image_items}

        return None

    def _api_headers(self, api_key: str, *, content_type: str | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {api_key}"}
        if content_type:
            headers["Content-Type"] = content_type

        organization = self._str_cfg("api", "organization", "")
        project = self._str_cfg("api", "project", "")
        if organization:
            headers["OpenAI-Organization"] = organization
        if project:
            headers["OpenAI-Project"] = project
        return headers

    def _api_key(self, provider_mode: str) -> str:
        if provider_mode == "2api":
            return (
                self._str_cfg("two_api", "api_key", "")
                or self._str_cfg("api", "api_key", "")
                or os.environ.get("TWO_API_KEY", "")
                or os.environ.get("OPENAI_API_KEY", "")
            )
        return (
            self._str_cfg("openai", "api_key", "")
            or self._str_cfg("api", "api_key", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )

    def _ensure_enabled(self) -> None:
        provider_mode = self._provider_mode()
        if provider_mode == "openai" and not self._provider_enabled("openai"):
            raise RuntimeError("OpenAI 官方接口没有启用。请在插件配置里开启 openai.enabled，或启用 2api。")
        if provider_mode == "2api" and not self._provider_enabled("2api"):
            raise RuntimeError("2api 接口没有启用。请在插件配置里开启 two_api.enabled，或启用 OpenAI 官方接口。")

    def _remove_tool_from_request(self, req: ProviderRequest) -> None:
        tool_set = req.func_tool
        if isinstance(tool_set, FunctionToolManager):
            req.func_tool = tool_set.get_full_tool_set()
            tool_set = req.func_tool
        if tool_set and hasattr(tool_set, "remove_tool"):
            tool_set.remove_tool(TOOL_NAME)

    def _cleanup_old_images(self) -> None:
        ttl_hours = self._int_cfg("runtime", "cache_ttl_hours", 72)
        max_files = self._int_cfg("runtime", "max_cache_files", 200)
        if ttl_hours <= 0 and max_files <= 0:
            return

        try:
            files = sorted(
                (path for path in self._image_dir.iterdir() if path.is_file()),
                key=lambda path: path.stat().st_mtime,
            )
            now = time.time()
            if ttl_hours > 0:
                expire_before = now - ttl_hours * 3600
                for path in files:
                    if path.stat().st_mtime < expire_before:
                        path.unlink(missing_ok=True)
                files = [path for path in files if path.exists()]
            if max_files > 0 and len(files) > max_files:
                for path in files[: len(files) - max_files]:
                    path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to clean old GPT Image cache files", exc_info=True)

    def _format_api_error(self, status: int, text: str) -> str:
        try:
            data = json.loads(text)
            error = data.get("error", data)
            if isinstance(error, dict):
                message = error.get("message") or error.get("code") or str(error)
            else:
                message = str(error)
        except Exception:
            message = text
        return f"图像接口请求失败：HTTP {status} {message[:600]}"

    def _friendly_error(self, exc: Exception) -> str:
        text = str(exc).strip()
        return text if text else exc.__class__.__name__

    def _section(self, section: str) -> dict[str, Any]:
        value = self.config.get(section, {}) if isinstance(self.config, dict) else {}
        return value if isinstance(value, dict) else {}

    def _str_cfg(self, section: str, key: str, default: str) -> str:
        value = self._section(section).get(key, default)
        if value is None:
            return default
        return str(value).strip()

    def _int_cfg(self, section: str, key: str, default: int) -> int:
        value = self._section(section).get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _bool_cfg(self, section: str, key: str, default: bool) -> bool:
        value = self._section(section).get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}
        return bool(value)
