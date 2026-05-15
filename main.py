"""AstrBot plugin for GPT image generation models.

The plugin exposes a single API config block. Internally it keeps two
parameter profiles (``standard`` and ``flexible``) that map to OpenAI's
strict Images API and the looser webpage-reverse / 2api / ToAPIs dialect.
A profile is picked per request from cache or heuristics; on a parameter
or vague upstream compatibility error the request is retried once with the
other profile, and the successful profile is cached per
``(base_url, model, operation)``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
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
from astrbot.api.star import Context, Star, StarTools, register


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
TRANSIENT_HTTP_STATUSES = {408, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526}
NETWORK_RETRY_BASE_DELAY_SECONDS = 1.5

OPENAI_SIZE_TO_RATIO = {
    "1024x1024": "1:1",
    "1536x1024": "3:2",
    "1024x1536": "2:3",
}
GPT_IMAGE_2_SIZE_TABLE = {
    "1k": {
        "1:1": "1024x1024",
        "3:2": "1536x1024",
        "2:3": "1024x1536",
        "4:3": "1024x768",
        "3:4": "768x1024",
        "5:4": "1280x1024",
        "4:5": "1024x1280",
        "16:9": "1536x864",
        "9:16": "864x1536",
        "2:1": "2048x1024",
        "1:2": "1024x2048",
        "21:9": "2016x864",
        "9:21": "864x2016",
    },
    "2k": {
        "1:1": "2048x2048",
        "3:2": "2048x1360",
        "2:3": "1360x2048",
        "4:3": "2048x1536",
        "3:4": "1536x2048",
        "5:4": "2560x2048",
        "4:5": "2048x2560",
        "16:9": "2048x1152",
        "9:16": "1152x2048",
        "2:1": "2688x1344",
        "1:2": "1344x2688",
        "21:9": "2688x1152",
        "9:21": "1152x2688",
    },
    "4k": {
        "16:9": "3840x2160",
        "9:16": "2160x3840",
        "2:1": "3840x1920",
        "1:2": "1920x3840",
        "21:9": "3840x1648",
        "9:21": "1648x3840",
    },
}

# Keyword sets used by ``ImageAPIError.should_try_other_profile`` to decide
# whether retrying once with the other compat profile is worth attempting.
#
# A profile swap helps when the upstream rejected our payload format. It does
# NOT help and would waste a request when the failure is auth, quota,
# moderation, or a missing model. The lists below are kept conservative so
# generic words (``policy`` alone, ``quota`` alone) do not accidentally swallow
# legitimate parameter errors that mention them in passing.
CONTENT_POLICY_KEYWORDS = (
    "content_policy",
    "content policy",
    "moderation",
    "safety system",
    "rejected by safety",
    "violates",
    "violate",
    "inappropriate",
)
BILLING_KEYWORDS = (
    "insufficient_quota",
    "insufficient quota",
    "billing",
    "exceeded your current quota",
    "exceeded your quota",
    "credit",
    "balance",
    "余额",
)
NON_PROFILE_ERROR_KEYWORDS = (
    "invalid api key",
    "incorrect api key",
    "unauthorized",
    "forbidden",
    "model not found",
    "model_not_found",
    "unknown model",
    "does not exist",
)

# Defaults for the merged ``api`` config block. Used by ``_new_api_active`` to
# tell whether the user has actually filled in the new block or is still on
# the auto-populated schema defaults (in which case we should fall back to
# the legacy ``openai`` / ``two_api`` blocks).
DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_MODEL = "gpt-image-2"
DEFAULT_API_TIMEOUT = 180
PROMPT_REWRITE_GUARD_PREFIX = "Use the following text as the complete prompt. Do not rewrite it:"

# aiohttp's default User-Agent looks like ``Python/3.x aiohttp/3.y``. Some
# middlemen place broad Cloudflare rules in front of API endpoints that block
# that generic UA before the request reaches the model server. Identify this
# plugin as a non-browser API client by default, while still allowing users to
# override the header with ``api.user_agent`` for stricter deployments.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; AstrBotGPTImagePlugin/1.0; "
    "+https://github.com/starmiaoa/astrbot-plugin-gpt-image)"
)

# Pattern that identifies Cloudflare anti-bot / browser-challenge / managed
# challenge HTML responses (typically returned with HTTP 403 and a ``cf-ray``
# header). Used to surface a clearer error than raw HTML to the user.
#
# CF Ray IDs are exactly 16 hex chars, optionally followed by an upper-case
# datacenter code like ``-LAX``. Anchor the search to ``cf-ray`` / ``Ray ID`` so
# we do not accidentally report an unrelated asset hash from the challenge page.
_CF_RAY_PATTERN = re.compile(
    r"(?:cf[-_ ]?ray|ray\s+id)[^0-9a-f]{0,120}([0-9a-f]{16}(?:-[A-Z]{2,4})?)",
    re.IGNORECASE,
)


class ImageAPIError(RuntimeError):
    """HTTP error returned from the image API.

    Wraps the status, parsed message, and (truncated) raw body so the
    caller can decide whether the failure is a parameter format issue
    that warrants a retry with the other compat profile.
    """

    def __init__(self, status: int, message: str, body: str = "") -> None:
        super().__init__(message or f"HTTP {status}")
        self.status = status
        self.message = message
        self.body = body

    def should_try_other_profile(self) -> bool:
        """Whether retrying once with the other compat profile is useful.

        Many OpenAI-compatible proxies return vague 5xx errors such as
        ``openai_error`` for payloads they cannot translate. We only use this
        on the first profile attempt, so a permissive answer costs at most one
        extra request before surfacing the real error.
        """
        if self.status in {401, 403, 404, 429}:
            return False
        haystack = f"{self.message}\n{self.body}".lower()
        if any(keyword in haystack for keyword in CONTENT_POLICY_KEYWORDS):
            return False
        if any(keyword in haystack for keyword in BILLING_KEYWORDS):
            return False
        if any(keyword in haystack for keyword in NON_PROFILE_ERROR_KEYWORDS):
            return False
        return self.status in {400, 422}


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
    "1.2.11",
)
class GPTImage2Plugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}
        self._data_dir = StarTools.get_data_dir()
        self._image_dir = self._data_dir / "images"
        self._image_dir.mkdir(parents=True, exist_ok=True)

        max_concurrent = self._int_cfg("runtime", "max_concurrent", 1)
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent))
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._inflight_tool_tasks: dict[str, asyncio.Task[tuple[str, str | None]]] = {}
        self._inflight_tool_turns: dict[str, asyncio.Task[tuple[str, str | None]]] = {}
        # In-memory cache of the compat profile that succeeded last time for a
        # given (base_url, model, operation). Lets the next request skip the
        # auto-detect heuristic and the failure-retry roundtrip.
        self._compat_profile_cache: dict[tuple[str, str, str], str] = {}

    @filter.command("生图")
    async def generate_command(self, event: AstrMessageEvent):
        prompt = self._extract_command_prompt(event, ("生图",))
        opts, prompt = self._parse_inline_options(prompt)
        if not prompt:
            yield event.plain_result(
                "用法：/生图 一只赛博猫坐在霓虹窗边 --size 1024x1024 --quality high"
            ).stop_event()
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
        self._schedule_background_delivery(event, task, error_prefix="图片生成失败")
        event.stop_event()
        yield event.plain_result(self._tool_submitted_message())

    @filter.command("改图")
    async def edit_command(self, event: AstrMessageEvent):
        prompt = self._extract_command_prompt(event, ("改图",))
        opts, prompt = self._parse_inline_options(prompt)
        if not prompt:
            yield event.plain_result(
                "用法：发送或引用图片后，输入 /改图 把背景换成夜晚城市 --size 1536x1024 --quality high"
            ).stop_event()
            return

        reference_paths = await self._collect_reference_image_paths(event)
        if not reference_paths:
            yield event.plain_result("没有找到可用参考图。请在同一条消息里发图，或回复/引用一张图后再使用 /改图。").stop_event()
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
        self._schedule_background_delivery(event, task, error_prefix="图片修改失败")
        event.stop_event()
        yield event.plain_result(self._tool_submitted_message())

    @filter.command("生图帮助")
    async def gptimage_help(self, event: AstrMessageEvent):
        yield event.plain_result(
            "GPT Image 图像生成插件已加载。\n"
            "命令：/生图 提示词；/改图 提示词（同消息发图，或回复/引用图片）。\n"
            "参数：--resolution 1k|2k|4k，--ratio 1:1|16:9|9:16|3:2|2:3，"
            "--size 1024x1024|1536x1024|1024x1536|auto，"
            "--quality auto|low|medium|high，--style 风格，--transparent。\n"
            "说明：像素尺寸用 --size，清晰度档位用 --resolution，比例用 --ratio。\n"
            f"多图参考：GPT 图像编辑接口最多支持 {MAX_GPT_IMAGE_REFERENCE_IMAGES} 张参考图。"
        ).stop_event()

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

        if not self._message_allows_image_tool(event, use_reference_images=use_reference_images):
            logger.info(
                "GPT Image tool call blocked because the source message has no explicit image intent: %s",
                self._event_text(event)[:200],
            )
            return "这条消息不是明确的出图或改图请求 不要调用图片工具 请直接用文字回复用户"

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

        try:
            await event.send(event.chain_result(self._build_result_chain(image_path, revised_prompt)))
        except Exception as exc:
            logger.error("GPT Image foreground delivery failed", exc_info=True)
            return f"图片生成失败：{self._friendly_error(exc)}"
        return None

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
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError(
                "图像接口没有配置 API Key。请填写插件配置里的 api.api_key，"
                "或设置环境变量 OPENAI_API_KEY / TWO_API_KEY。"
            )

        references = self._prepare_reference_image_paths(reference_image_paths or [])
        final_prompt = self._build_prompt(prompt, style)

        operation = "edit" if references else "generation"
        base_url = self._base_url()
        model = self._model_name()
        cache_key = (base_url, model, operation)

        cached_profile = self._compat_profile_cache.get(cache_key)
        heuristic_profile = self._auto_detect_profile(
            operation=operation,
            size=size,
            aspect_ratio=aspect_ratio,
            reference_image_paths=references,
        )
        if self._request_has_strong_profile_hint(
            operation=operation,
            size=size,
            aspect_ratio=aspect_ratio,
            reference_image_paths=references,
        ):
            initial_profile = heuristic_profile
        else:
            initial_profile = cached_profile or heuristic_profile
        other_profile = "flexible" if initial_profile == "standard" else "standard"
        attempts = [initial_profile, other_profile]

        errors: list[ImageAPIError] = []
        response: dict[str, Any] | None = None
        succeeded_profile: str | None = None

        async with self._semaphore:
            for idx, profile in enumerate(attempts):
                payload = self._build_payload(
                    final_prompt,
                    profile=profile,
                    size=size,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    quality=quality,
                    transparent_background=transparent_background,
                    reference_image_paths=references,
                )
                self._log_request_start(
                    operation=operation,
                    profile=profile,
                    payload=payload,
                )
                try:
                    if references:
                        response = await self._post_images_edit_api(api_key, payload, references)
                    else:
                        response = await self._post_images_generation_api(api_key, payload)
                except ImageAPIError as exc:
                    errors.append(exc)
                    logger.warning(
                        "GPT Image %s with profile=%s failed: HTTP %d %s",
                        operation,
                        profile,
                        exc.status,
                        str(exc.message)[:200],
                    )
                    if idx == 0 and exc.should_try_other_profile():
                        # Worth a single retry with the other parameter dialect.
                        continue
                    break
                else:
                    succeeded_profile = profile
                    break

            if response is None:
                if not errors:
                    raise RuntimeError("图像接口请求失败：未知错误。")
                if len(errors) > 1:
                    logger.error(
                        "GPT Image both profiles failed; first profile error HTTP %d: %s",
                        errors[0].status,
                        str(errors[0].message)[:300],
                    )
                last = errors[-1]
                raise RuntimeError(
                    self._format_api_error(last.status, last.body or last.message)
                ) from last

            if succeeded_profile:
                self._compat_profile_cache[cache_key] = succeeded_profile
                if cached_profile and cached_profile != succeeded_profile:
                    logger.info(
                        "GPT Image profile updated for %s | %s | %s: %s -> %s",
                        base_url,
                        model,
                        operation,
                        cached_profile,
                        succeeded_profile,
                    )

            image_path, revised_prompt = await self._save_image_from_response(response)
            self._cleanup_old_images()
            return image_path, revised_prompt

    async def _post_images_generation_api(
        self,
        api_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        url = self._images_generation_url()
        headers = self._api_headers(api_key, content_type="application/json")
        timeout = aiohttp.ClientTimeout(total=max(10, self._timeout_seconds()))
        retry_times = self._mutation_retry_times()

        for attempt in range(retry_times + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                    async with session.post(url, headers=headers, json=payload) as resp:
                        if resp.status in TRANSIENT_HTTP_STATUSES and attempt < retry_times:
                            text = await self._read_limited_response_text(resp)
                            await self._sleep_before_network_retry(
                                attempt,
                                operation="GPT Image generation",
                                reason=f"HTTP {resp.status} {text[:200]}",
                            )
                            continue
                        if resp.status >= 400:
                            text = await self._read_limited_response_text(resp)
                            raise self._build_api_error(resp.status, text)
                        text = await resp.text()
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError as exc:
                            raise RuntimeError("图像接口返回了无法解析的 JSON。") from exc
            except self._transient_network_exceptions() as exc:
                if attempt < retry_times:
                    await self._sleep_before_network_retry(
                        attempt,
                        operation="GPT Image generation",
                        reason=self._friendly_error(exc),
                    )
                    continue
                raise RuntimeError(self._network_failure_message(exc, "图像接口请求失败")) from exc

        raise RuntimeError("图像接口请求失败：网络重试后仍未返回结果。")

    async def _post_images_edit_api(
        self,
        api_key: str,
        payload: dict[str, Any],
        image_paths: list[str],
    ) -> dict[str, Any]:
        url = self._images_edit_url()
        headers = self._api_headers(api_key)
        timeout = aiohttp.ClientTimeout(total=max(10, self._timeout_seconds()))
        retry_times = self._mutation_retry_times()

        for attempt in range(retry_times + 1):
            form = aiohttp.FormData()
            for key, value in payload.items():
                form.add_field(key, self._multipart_field_value(value))

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

                try:
                    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                        async with session.post(url, headers=headers, data=form) as resp:
                            if resp.status in TRANSIENT_HTTP_STATUSES and attempt < retry_times:
                                text = await self._read_limited_response_text(resp)
                                await self._sleep_before_network_retry(
                                    attempt,
                                    operation="GPT Image edit",
                                    reason=f"HTTP {resp.status} {text[:200]}",
                                )
                                continue
                            if resp.status >= 400:
                                text = await self._read_limited_response_text(resp)
                                raise self._build_api_error(resp.status, text)
                            text = await resp.text()
                            try:
                                return json.loads(text)
                            except json.JSONDecodeError as exc:
                                raise RuntimeError("图像编辑接口返回了无法解析的 JSON。") from exc
                except self._transient_network_exceptions() as exc:
                    if attempt < retry_times:
                        await self._sleep_before_network_retry(
                            attempt,
                            operation="GPT Image edit",
                            reason=self._friendly_error(exc),
                        )
                        continue
                    raise RuntimeError(self._network_failure_message(exc, "图像编辑接口请求失败")) from exc
            finally:
                for file_obj in opened_files:
                    try:
                        file_obj.close()
                    except Exception:
                        logger.debug("Failed to close GPT Image upload file", exc_info=True)

        raise RuntimeError("图像编辑接口请求失败：网络重试后仍未返回结果。")

    async def _save_image_from_response(self, response: dict[str, Any]) -> tuple[str, str | None]:
        normalized_response = self._extract_image_response(response)
        if normalized_response:
            response = normalized_response

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
            try:
                image_bytes = base64.b64decode(raw_b64)
            except (binascii.Error, ValueError) as exc:
                raise RuntimeError("图像接口返回的 b64_json 无法解码。") from exc
            if not image_bytes:
                raise RuntimeError("图像接口返回了空的 b64_json。")
            actual_suffix = self._image_suffix_from_bytes(image_bytes)
            if actual_suffix:
                file_path = file_path.with_suffix(actual_suffix)
            file_path.write_bytes(image_bytes)
            return str(file_path), revised_prompt

        image_url = item.get("url")
        if isinstance(image_url, str) and image_url.startswith(("http://", "https://")):
            file_path = await self._download_image(image_url, file_path)
            return str(file_path), revised_prompt

        raise RuntimeError("图像接口没有返回 b64_json 或 url。")

    async def _download_image(self, url: str, file_path: Path) -> Path:
        timeout = aiohttp.ClientTimeout(total=max(10, self._timeout_seconds()))
        retry_times = self._network_retry_times()
        headers = {"User-Agent": self._user_agent()}

        for attempt in range(retry_times + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                    async with session.get(url, headers=headers) as resp:
                        if resp.status in TRANSIENT_HTTP_STATUSES and attempt < retry_times:
                            text = await self._read_limited_response_text(resp)
                            await self._sleep_before_network_retry(
                                attempt,
                                operation="GPT Image download",
                                reason=f"HTTP {resp.status} {text[:200]}",
                            )
                            continue
                        if resp.status >= 400:
                            text = await self._read_limited_response_text(resp)
                            cf_message = self._cf_challenge_message(resp.status, text)
                            if cf_message:
                                raise RuntimeError(cf_message)
                            raise RuntimeError(f"下载图片失败：HTTP {resp.status} {text[:300]}")
                        data = await resp.read()
                        if not data:
                            raise RuntimeError("下载图片失败：响应内容为空。")
                        actual_suffix = self._image_suffix_from_bytes(data, resp.headers.get("Content-Type", ""))
                        if actual_suffix:
                            file_path = file_path.with_suffix(actual_suffix)
                        file_path.write_bytes(data)
                        return file_path
            except self._transient_network_exceptions() as exc:
                if attempt < retry_times:
                    await self._sleep_before_network_retry(
                        attempt,
                        operation="GPT Image download",
                        reason=self._friendly_error(exc),
                    )
                    continue
                raise RuntimeError(self._network_failure_message(exc, "下载图片失败")) from exc

        raise RuntimeError("下载图片失败：网络重试后仍未返回结果。")

    async def _read_limited_response_text(self, resp: aiohttp.ClientResponse, limit: int = 2048) -> str:
        raw = await resp.content.read(limit + 1)
        truncated = len(raw) > limit
        if truncated:
            raw = raw[:limit]
        text = raw.decode(resp.charset or "utf-8", errors="replace")
        return f"{text}..." if truncated else text

    def _multipart_field_value(self, value: Any) -> str:
        if value is True:
            return "true"
        if value is False:
            return "false"
        return str(value)

    def _network_retry_times(self) -> int:
        return min(5, max(0, self._int_cfg("runtime", "retry_times", 1)))

    def _mutation_retry_times(self) -> int:
        # Image generation/edit requests are side-effecting POSTs. If the
        # upstream accepts the job but the client times out before receiving the
        # response, retrying can create a second paid image task. Until the
        # supplier exposes a reliable idempotency key, keep POST retries off.
        return 0

    def _log_request_start(self, *, operation: str, profile: str, payload: dict[str, Any]) -> None:
        url = self._images_edit_url() if operation == "edit" else self._images_generation_url()
        logger.info(
            "GPT Image request start operation=%s profile=%s model=%s url=%s timeout=%s retry=%s size=%s resolution=%s",
            operation,
            profile,
            payload.get("model", ""),
            url,
            self._timeout_seconds(),
            self._mutation_retry_times(),
            payload.get("size", ""),
            payload.get("resolution", ""),
        )

    def _transient_network_exceptions(self) -> tuple[type[BaseException], ...]:
        return (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError, ConnectionError)

    async def _sleep_before_network_retry(self, attempt: int, *, operation: str, reason: str) -> None:
        delay = min(8.0, NETWORK_RETRY_BASE_DELAY_SECONDS * (2 ** attempt))
        logger.warning(
            "%s transient network failure; retrying in %.1fs (%d/%d): %s",
            operation,
            delay,
            attempt + 1,
            self._network_retry_times(),
            reason[:200],
        )
        await asyncio.sleep(delay)

    def _schedule_background_delivery(
        self,
        event: AstrMessageEvent,
        task: asyncio.Task[tuple[str, str | None]],
        *,
        error_prefix: str = "图片生成失败",
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
                    await event.send(event.plain_result(f"{error_prefix}：{self._friendly_error(exc)}"))
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
        profile: str,
        size: str | None,
        aspect_ratio: str | None,
        resolution: str | None,
        quality: str | None,
        transparent_background: bool,
        reference_image_paths: list[str],
    ) -> dict[str, Any]:
        model = self._model_name()
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": 1,
        }

        # ``standard`` keeps the OpenAI-strict size grammar (pixel sizes only,
        # ratios mapped to one of three OpenAI-accepted dimensions). ``flexible``
        # lets the size field carry an aspect ratio string the way most webpage
        # reverse / 2api suppliers expect.
        if profile == "flexible":
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
                resolution=resolution,
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
            if (
                normalized_resolution == "4k"
                and normalized_ratio not in VALID_4K_ASPECT_RATIOS
                and (profile == "flexible" or self._model_supports_gpt_image_2_sizes())
            ):
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
        guard = self._bool_cfg("prompt", "prevent_prompt_rewrite", True)

        if prefix:
            parts.append(prefix.strip())
        user_prompt = prompt.strip()
        if guard and user_prompt:
            user_prompt = f"{PROMPT_REWRITE_GUARD_PREFIX}\n{user_prompt}"
        parts.append(user_prompt)
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
                logger.debug("Failed to read AstrBot message id", exc_info=True)
                continue
            if value:
                parts.append(str(value))

        text = ""
        try:
            text = str(getattr(event, "message_str", "") or event.get_message_str() or "")
        except Exception:
            logger.debug("Failed to read AstrBot message text for image tool key", exc_info=True)
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
                logger.debug("Failed to read AstrBot event scope", exc_info=True)
                continue
            if value:
                parts.append(str(value))

        return "|".join(dict.fromkeys(parts)) or "global"

    def _message_allows_image_tool(self, event: AstrMessageEvent, *, use_reference_images: bool) -> bool:
        text = self._event_text(event)
        if not text:
            return False

        normalized = re.sub(r"\s+", "", text.lower())
        if not normalized:
            return False

        prompt_only_patterns = [
            r"(提示词|prompt).*(写|改|优化|润色|翻译|扩写|整理)",
            r"(生成|写|改|优化|润色|翻译|扩写|整理).*(提示词|prompt)",
            r"(怎么|如何|为什么|什么|啥|能不能|可以吗|参数|设置|教程|建议|方案|分析|评价).*(画|生成|生图|出图|改图|图片|图像)",
            r"(how|what|why|help|guide|tutorial|advice|suggest|analy[sz]e|review).*(draw|generate|create|edit|image|picture|photo|prompt)",
            r"(prompt).*(write|improve|rewrite|translate|optimi[sz]e|polish)",
            r"(write|improve|rewrite|translate|optimi[sz]e|polish).*(prompt)",
        ]
        if any(re.search(pattern, normalized) for pattern in prompt_only_patterns):
            return False

        generation_patterns = [
            r"(画|绘制|生成|生图|出图|出一张|做一张|做张|做一个|做个|来一张|来张).*(图|图片|图像|画|海报|头像|壁纸|图标|logo|贴纸|表情|插画|漫画|照片|场景)",
            r"(画|绘制)(一张|一个|个|只|幅|张).+",
            r"(生成|做|设计|制作)(一张|一个|个|张).+",
            r"(帮我|给我|替我).*(画|绘制|生成|生图|出图|做一张|做张|做一个|做个|来一张|来张)",
            r"(设计|制作|做).*(一张|一个|个).*(图|图片|图像|海报|头像|壁纸|图标|logo|贴纸|表情|插画|漫画)",
            r"^(画|绘制|生成|生图|出图|做一张|做张|做一个|做个|来一张|来张)",
            r"(draw|generate|create|make|paint|design).*(image|picture|photo|poster|avatar|wallpaper|icon|logo|sticker|illustration|comic|scene)",
            r"(image|picture|photo|poster|avatar|wallpaper|icon|logo|sticker|illustration).*(draw|generate|create|make|paint|design)",
            r"(can|could|please|pls|wouldyou|canyou|couldyou|helpme).*(draw|paint).+",
            r"(can|could|please|pls|wouldyou|canyou|couldyou|helpme).*(generate|create|make|design).*(image|picture|photo|poster|avatar|wallpaper|icon|logo|sticker|illustration|comic|scene)",
            r"^(draw|paint)(a|an|the)?.+",
            r"^(draw|generate|create|make|paint|design)(an?|the)?(image|picture|photo|poster|avatar|wallpaper|icon|logo|sticker|illustration|comic|scene)",
            r"(画像|イラスト|写真|ポスター|アイコン|壁紙|ロゴ).*(描いて|生成|作成|作って|描く)",
            r"(描いて|生成|作成|作って).*(画像|イラスト|写真|ポスター|アイコン|壁紙|ロゴ)",
            r"(描いて|描く|生成して|作って|作成して)",
        ]
        edit_patterns = [
            r"(改图|修图|p图|重绘|扩图|抠图)",
            r"(改|修|换|替换|去掉|删除|加上|添加|合成|参考|照着).*(这张|上图|图片|图里|图中|照片|背景|主体|人物|文字)",
            r"(把|将).*(图|图片|照片|背景|主体|人物|文字).*(改|修|换|替换|去掉|删除|加上|添加|合成)",
            r"(edit|modify|change|replace|remove|delete|add|combine|merge|redraw|expand|cutout).*(this|the)?(image|picture|photo|background|subject|person|text)",
            r"(this|the)?(image|picture|photo|background|subject|person|text).*(edit|modify|change|replace|remove|delete|add|combine|merge|redraw|expand|cutout)",
            r"(画像|写真|背景|人物|文字).*(編集|修正|変更|置換|削除|追加|合成)",
        ]

        if use_reference_images:
            return any(re.search(pattern, normalized) for pattern in edit_patterns + generation_patterns)
        return any(re.search(pattern, normalized) for pattern in generation_patterns)

    def _event_text(self, event: AstrMessageEvent) -> str:
        try:
            return str(getattr(event, "message_str", "") or event.get_message_str() or "").strip()
        except Exception:
            logger.debug("Failed to read AstrBot message text", exc_info=True)
            return str(getattr(event, "message_str", "") or "").strip()

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

        prompt = " ".join(prompt_parts).strip()
        self._extract_natural_language_options(opts, prompt)
        return opts, prompt

    def _extract_natural_language_options(self, opts: dict[str, Any], prompt: str) -> None:
        if not prompt:
            return
        if not opts.get("aspect_ratio"):
            ratio = self._extract_ratio_from_text(prompt)
            if ratio:
                opts["aspect_ratio"] = ratio
        if not opts.get("resolution"):
            resolution = self._extract_resolution_from_text(prompt)
            if resolution:
                opts["resolution"] = resolution

    def _assign_size_like_option(self, opts: dict[str, Any], value: str) -> None:
        normalized = (value or "").strip().lower()
        if normalized == "auto":
            opts["size"] = "auto"
            return
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

    def _extract_ratio_from_text(self, text: str) -> str:
        for match in re.finditer(r"(?<!\d)(\d{1,2})\s*[:：]\s*(\d{1,2})(?!\d)", text):
            start = max(0, match.start() - 16)
            end = min(len(text), match.end() + 16)
            context = text[start:end].lower()
            if any(keyword in context for keyword in ("不要当比例", "不是比例", "非比例", "not ratio", "not aspect")):
                continue
            if not any(
                keyword in context
                for keyword in (
                    "比例",
                    "宽高比",
                    "寬高比",
                    "画幅",
                    "畫幅",
                    "横竖比",
                    "纵横比",
                    "ratio",
                    "aspect",
                    "aspect ratio",
                )
            ):
                continue
            ratio = self._parse_aspect_ratio(f"{match.group(1)}:{match.group(2)}")
            if ratio:
                return ratio
        return ""

    def _extract_resolution_from_text(self, text: str) -> str:
        lowered = text.lower()
        for match in re.finditer(r"(?<![a-z0-9])([124])\s*k(?![a-z0-9])", lowered):
            start = max(0, match.start() - 12)
            end = min(len(lowered), match.end() + 12)
            context = lowered[start:end]
            if any(keyword in context for keyword in ("分辨率", "清晰度", "resolution", "画质", "畫質")):
                return f"{match.group(1)}k"
        return ""

    def _extract_command_prompt(self, event: AstrMessageEvent, command_names: tuple[str, ...]) -> str:
        text = self._event_text(event)
        for name in command_names:
            for prefix in (f"/{name}", name):
                if text == prefix:
                    return ""
                if text.startswith(prefix):
                    return text[len(prefix) :].lstrip()
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
            logger.debug("Failed to read AstrBot message chain", exc_info=True)
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
        except OSError:
            logger.debug("Failed to read image header for content type detection", exc_info=True)
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
        resolution: str | None,
        reference_image_paths: list[str],
        prefer_reference_ratio: bool,
    ) -> str:
        explicit_size = self._normalize_pixel_size(size)
        if explicit_size:
            return explicit_size

        ratio = self._normalize_aspect_ratio(
            aspect_ratio,
            size=size,
            reference_image_paths=reference_image_paths,
            prefer_reference_ratio=prefer_reference_ratio,
        )
        if ratio != "auto":
            gpt_image_size = self._gpt_image_2_size_for_ratio(
                ratio,
                self._normalize_resolution(resolution),
            )
            if gpt_image_size:
                return gpt_image_size
            return self._openai_size_for_ratio(ratio)

        configured_size = self._normalize_pixel_size(self._str_cfg("image", "size", "1024x1024"))
        if configured_size:
            return configured_size
        return "1024x1024"

    def _normalize_pixel_size(self, size: str | None) -> str:
        value = (size or "").strip().lower()
        if not value or value == "auto":
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
            return ""
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
            return ""
        width, height = size
        if width <= 0 or height <= 0:
            return ""
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

    def _gpt_image_2_size_for_ratio(self, ratio: str, resolution: str) -> str:
        if not self._model_supports_gpt_image_2_sizes():
            return ""
        tier = resolution if resolution in VALID_RESOLUTIONS else "auto"
        if tier == "auto":
            tier = self._normalize_resolution(None)
        if tier == "auto":
            return ""
        sizes = GPT_IMAGE_2_SIZE_TABLE.get(tier, {})
        if ratio in sizes:
            return sizes[ratio]
        if tier == "4k":
            return GPT_IMAGE_2_SIZE_TABLE["2k"].get(ratio, "")
        return ""

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
            value = self._str_cfg("image", "resolution", "1k").lower()
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
        except (ImportError, OSError, ValueError):
            logger.debug("Failed to read image dimensions with Pillow", exc_info=True)

        try:
            with open(image_path, "rb") as file_obj:
                data = file_obj.read(256 * 1024)
        except OSError:
            logger.debug("Failed to read image header for dimension detection", exc_info=True)
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

    def _images_generation_url(self) -> str:
        base_url = self._base_url().rstrip("/")
        if base_url.endswith("/images/generations"):
            return base_url
        if base_url.endswith("/images/edits"):
            return f"{base_url.rsplit('/images/', 1)[0]}/images/generations"
        if base_url.endswith("/v1"):
            return f"{base_url}/images/generations"
        return f"{base_url}/v1/images/generations"

    def _images_edit_url(self) -> str:
        base_url = self._base_url().rstrip("/")
        if base_url.endswith("/images/edits"):
            return base_url
        if base_url.endswith("/images/generations"):
            return f"{base_url.rsplit('/images/', 1)[0]}/images/edits"
        if base_url.endswith("/v1"):
            return f"{base_url}/images/edits"
        return f"{base_url}/v1/images/edits"

    def _new_api_active(self) -> bool:
        """Whether the merged ``api`` config block is filled in by the user.

        Schema defaults populate ``api.base_url`` / ``api.model`` /
        ``api.timeout_seconds`` automatically, so we cannot rely solely on
        them being present. We treat the new block as active when:

        * the user has set ``api.api_key`` (most explicit signal); or
        * any of ``base_url`` / ``model`` / ``timeout_seconds`` differs from
          the schema defaults (the user has clearly customized the new
          block even though they leave the key in an env var).

        This avoids the bug where a legacy ``openai`` block silently wins
        over a freshly customized ``api.base_url`` when the key only lives
        in ``OPENAI_API_KEY`` / ``TWO_API_KEY``.
        """
        if self._str_cfg("api", "api_key", ""):
            return True

        section = self._section("api")
        base_url = self._str_cfg("api", "base_url", "")
        if base_url and base_url != DEFAULT_API_BASE_URL:
            return True
        model = self._str_cfg("api", "model", "")
        if model and model != DEFAULT_API_MODEL:
            return True
        raw_timeout = section.get("timeout_seconds")
        if raw_timeout is not None:
            try:
                if int(raw_timeout) != DEFAULT_API_TIMEOUT:
                    return True
            except (TypeError, ValueError):
                pass
        return False

    def _legacy_preferred_section(self) -> str:
        """Pick the legacy block the user was actively using.

        Defaults match the previous schema (``openai.enabled=true``,
        ``two_api.enabled=false``). When the user explicitly flipped to 2api,
        we honor that.
        """
        two_api_enabled = self._bool_cfg("two_api", "enabled", False)
        openai_enabled = self._bool_cfg("openai", "enabled", True)
        if two_api_enabled and not openai_enabled:
            return "two_api"
        return "openai"

    def _legacy_section_enabled(self, section: str) -> bool:
        if section == "two_api":
            return self._bool_cfg("two_api", "enabled", False)
        return self._bool_cfg("openai", "enabled", True)

    def _legacy_lookup(self, key: str) -> str:
        """Walk the legacy ``openai`` / ``two_api`` blocks for a string value.

        Respects the legacy ``enabled`` toggles so a user who explicitly
        disabled a block does not have its values silently picked up.
        """
        preferred = self._legacy_preferred_section()
        order = [preferred] + [s for s in ("openai", "two_api") if s != preferred]
        for section in order:
            if not self._legacy_section_enabled(section):
                continue
            value = self._str_cfg(section, key, "")
            if value:
                return value
        return ""

    def _legacy_lookup_int(self, key: str) -> int | None:
        preferred = self._legacy_preferred_section()
        order = [preferred] + [s for s in ("openai", "two_api") if s != preferred]
        for section in order:
            if not self._legacy_section_enabled(section):
                continue
            raw = self._section(section).get(key)
            if raw is None:
                continue
            try:
                return int(raw)
            except (TypeError, ValueError):
                continue
        return None

    def _model_name(self) -> str:
        if self._new_api_active():
            return self._str_cfg("api", "model", DEFAULT_API_MODEL) or DEFAULT_API_MODEL
        legacy = self._legacy_lookup("model")
        if legacy:
            return legacy
        # Super-old config block (pre openai/two_api split) reused the same
        # ``api`` section name as the new merged config.
        return self._str_cfg("api", "model", DEFAULT_API_MODEL) or DEFAULT_API_MODEL

    def _base_url(self) -> str:
        if self._new_api_active():
            return self._str_cfg("api", "base_url", DEFAULT_API_BASE_URL) or DEFAULT_API_BASE_URL
        legacy = self._legacy_lookup("base_url")
        if legacy:
            return legacy
        return self._str_cfg("api", "base_url", DEFAULT_API_BASE_URL) or DEFAULT_API_BASE_URL

    def _timeout_seconds(self) -> int:
        if self._new_api_active():
            return self._int_cfg("api", "timeout_seconds", DEFAULT_API_TIMEOUT)
        legacy = self._legacy_lookup_int("timeout_seconds")
        if legacy is not None:
            return legacy
        return self._int_cfg("api", "timeout_seconds", DEFAULT_API_TIMEOUT)

    def _auto_detect_profile(
        self,
        *,
        operation: str,
        size: str | None,
        aspect_ratio: str | None,
        reference_image_paths: list[str] | None,
    ) -> str:
        """Pick an initial compat profile when the cache misses.

        The returned value is only a first guess; if the upstream rejects
        the request with a parameter-format error the caller will retry once
        with the other profile.
        """
        # An explicit pixel size unambiguously speaks the strict OpenAI
        # dialect, so try ``standard`` first.
        if self._normalize_pixel_size(size):
            return "standard"

        # Webpage-reverse models often advertise broader aspect ratios through
        # the ``size`` field. Standard OpenAI-compatible endpoints commonly
        # reject ``size=16:9`` and expect a pixel size instead, so only start
        # with the flexible dialect when the model name clearly looks like a
        # reverse/web profile.
        explicit_ratio = self._parse_aspect_ratio(aspect_ratio) or self._parse_aspect_ratio(size)
        if explicit_ratio and explicit_ratio not in {"1:1", "3:2", "2:3"} and self._model_prefers_flexible_profile():
            return "flexible"

        # Image edits with reference images and no explicit ratio: flexible
        # can keep the original ratio without rounding to one of three sizes.
        if operation == "edit" and reference_image_paths and self._model_prefers_flexible_profile():
            return "flexible"

        return "standard"

    def _request_has_strong_profile_hint(
        self,
        *,
        operation: str,
        size: str | None,
        aspect_ratio: str | None,
        reference_image_paths: list[str] | None,
    ) -> bool:
        """Whether this request should override the cached compat profile.

        Cache is useful for ambiguous defaults, but an explicit pixel size,
        an explicit ratio request, or an edit request that should follow the
        reference image for a known-flexible model is a stronger signal than
        whatever succeeded on an earlier request.
        """
        if self._normalize_pixel_size(size):
            return True
        explicit_ratio = self._parse_aspect_ratio(aspect_ratio) or self._parse_aspect_ratio(size)
        if explicit_ratio:
            return True
        if operation == "edit" and reference_image_paths:
            return self._model_prefers_flexible_profile()
        return False

    def _model_prefers_flexible_profile(self) -> bool:
        """Whether the configured model name looks like a webpage-reverse SKU."""
        model = self._model_name().lower()
        flexible_markers = (
            "-all",
            "_all",
            "official",
            "toapi",
            "2api",
            "reverse",
            "web",
        )
        if any(marker in model for marker in flexible_markers):
            return True
        return model in {"gpt-image-1.5", "gpt-image-1.5-official"}

    def _model_supports_gpt_image_2_sizes(self) -> bool:
        model = self._model_name().strip().lower()
        return model == "gpt-image-2" or model.startswith("gpt-image-2-20")

    def _build_api_error(self, status: int, body: str) -> ImageAPIError:
        try:
            data = json.loads(body)
            error = data.get("error", data) if isinstance(data, dict) else data
            if isinstance(error, dict):
                message = error.get("message") or error.get("code") or str(error)
            else:
                message = str(error)
        except (json.JSONDecodeError, AttributeError, TypeError):
            message = body
        return ImageAPIError(status, str(message)[:600], body)

    def _extract_image_response(self, data: Any) -> dict[str, Any] | None:
        if isinstance(data, dict):
            payload_data = data.get("data")
            if isinstance(payload_data, list) and payload_data:
                # Normalize alias keys (``image_url`` / ``base64``) used by
                # webpage-reverse / 2api-style providers so the saver only
                # has to look at ``url`` / ``b64_json``.
                normalized_items: list[dict[str, Any]] = []
                for item in payload_data:
                    if not isinstance(item, dict):
                        continue
                    url = self._normalize_response_url(item.get("url") or item.get("image_url"))
                    b64 = item.get("b64_json") or item.get("base64")
                    if url or b64:
                        normalized_items.append({
                            "url": url,
                            "b64_json": b64,
                            "revised_prompt": item.get("revised_prompt"),
                        })
                if normalized_items:
                    return {"data": normalized_items}

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

            url = self._normalize_response_url(data.get("url") or data.get("image_url"))
            if url:
                return {"data": [{"url": url, "revised_prompt": data.get("revised_prompt")}]}
            b64_json = data.get("b64_json") or data.get("base64")
            if isinstance(b64_json, str) and b64_json.strip():
                return {"data": [{"b64_json": b64_json, "revised_prompt": data.get("revised_prompt")}]}

        if isinstance(data, list):
            image_items = []
            for item in data:
                if isinstance(item, str) and item.startswith(("http://", "https://")):
                    image_items.append({"url": item})
                elif isinstance(item, dict) and (
                    item.get("url") or item.get("b64_json") or item.get("image_url") or item.get("base64")
                ):
                    url = self._normalize_response_url(item.get("url") or item.get("image_url"))
                    b64 = item.get("b64_json") or item.get("base64")
                    if url or b64:
                        image_items.append(
                            {
                                "url": url,
                                "b64_json": b64,
                                "revised_prompt": item.get("revised_prompt"),
                            }
                        )
            if image_items:
                return {"data": image_items}

        return None

    def _normalize_response_url(self, value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("url")
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
        return ""

    def _image_suffix_from_bytes(self, data: bytes, content_type: str = "") -> str:
        content_type = (content_type or "").split(";", 1)[0].strip().lower()
        if content_type == "image/png" or data.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if content_type in {"image/jpeg", "image/jpg"} or data.startswith(b"\xff\xd8"):
            return ".jpg"
        if content_type == "image/webp" or (data[:4] == b"RIFF" and data[8:12] == b"WEBP"):
            return ".webp"
        return ""

    def _api_headers(self, api_key: str, *, content_type: str | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": self._user_agent(),
        }
        if content_type:
            headers["Content-Type"] = content_type

        organization = self._str_cfg("api", "organization", "")
        project = self._str_cfg("api", "project", "")
        if organization:
            headers["OpenAI-Organization"] = organization
        if project:
            headers["OpenAI-Project"] = project
        return headers

    def _user_agent(self) -> str:
        # Allow per-deployment override for middlemen that expect a specific
        # UA. Empty (the default) falls back to ``DEFAULT_USER_AGENT``.
        configured = self._str_cfg("api", "user_agent", "").strip()
        return configured or DEFAULT_USER_AGENT

    def _api_key(self) -> str:
        # New unified ``api`` block first. The same section name was used by
        # the very old pre-split config, so super-old users keep working too.
        new_key = self._str_cfg("api", "api_key", "")
        if new_key:
            return new_key

        # Legacy fallback only applies when the user is still on the old
        # blocks. Once they have customized the new ``api`` block (e.g. a
        # custom ``base_url``) and intentionally left ``api.api_key`` empty
        # to use an environment variable, a stale leftover key in the old
        # ``openai`` / ``two_api`` block must NOT silently be sent to the
        # new endpoint. ``_new_api_active`` already encapsulates this check.
        if not self._new_api_active():
            legacy_key = self._legacy_lookup("api_key")
            if legacy_key:
                return legacy_key

        # Last resort: environment variables. Order depends on which legacy
        # section the user was on so old 2api-mode users keep getting
        # ``TWO_API_KEY`` first when both vars happen to be set.
        if self._legacy_preferred_section() == "two_api":
            return os.environ.get("TWO_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
        return os.environ.get("OPENAI_API_KEY", "") or os.environ.get("TWO_API_KEY", "")

    def _ensure_enabled(self) -> None:
        # ``_api_key`` already walks the new ``api`` block, the legacy
        # ``openai`` / ``two_api`` blocks (honoring their ``enabled`` flags),
        # and the environment variables. An empty result means nothing usable
        # is configured anywhere.
        if not self._api_key():
            raise RuntimeError(
                "图像接口没有配置 API Key。"
                "请在插件配置里填写 api.api_key，或设置环境变量 OPENAI_API_KEY / TWO_API_KEY。"
            )

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
        cf_message = self._cf_challenge_message(status, text)
        if cf_message:
            return cf_message
        try:
            data = json.loads(text)
            error = data.get("error", data)
            if isinstance(error, dict):
                message = error.get("message") or error.get("code") or str(error)
            else:
                message = str(error)
        except (json.JSONDecodeError, AttributeError, TypeError):
            message = text
        return f"图像接口请求失败：HTTP {status} {message[:600]}"

    def _cf_challenge_message(self, status: int, text: str) -> str | None:
        # Cloudflare's anti-bot/managed-challenge pages are returned as HTML
        # with a ``cf-ray`` marker. Surface them as a clear, actionable error
        # instead of dumping raw HTML to the user.
        if status not in {403, 503} or not text:
            return None
        sample = text[:4000]
        lowered = sample.lower()
        if (
            "cf-ray" not in lowered
            and "cf_ray" not in lowered
            and "ray id" not in lowered
            and "cloudflare" not in lowered
        ):
            return None
        if not any(
            marker in lowered
            for marker in (
                "<html",
                "<!doctype html",
                "challenge",
                "just a moment",
                "attention required",
                "安全验证",
                "验证页",
                "cf-ray",
                "cf_ray",
            )
        ):
            return None
        ray_match = _CF_RAY_PATTERN.search(sample)
        ray_part = f"，cf-ray={ray_match.group(1)}" if ray_match else ""
        return (
            f"图像接口被 Cloudflare 拦截 (HTTP {status}{ray_part})。"
            "上游中转启用了 anti-bot 防护：可在插件配置 api.user_agent 里换一个 UA，"
            "或联系中转方放行你的出口 IP。"
        )

    def _friendly_error(self, exc: Exception) -> str:
        text = str(exc).strip()
        return text if text else exc.__class__.__name__

    def _network_failure_message(self, exc: BaseException, action: str) -> str:
        # Timeout-specific message: bare ``TimeoutError`` tells the user nothing
        # actionable. Surface the actual configured timeout so they know which
        # knob to turn.
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return (
                f"{action}：等待上游响应 {self._timeout_seconds()} 秒后超时。"
                "图像生成本身较慢，可在配置里调高 api.timeout_seconds（建议 180-300），"
                "或检查中转/网络是否可达。"
            )
        return f"{action}：网络连接失败：{self._friendly_error(exc)}"

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
