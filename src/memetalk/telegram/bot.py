from __future__ import annotations

import io
import logging
from collections.abc import MutableMapping
from typing import Any

from memetalk.app.container import AppContainer, build_container
from memetalk.config import AppSettings
from memetalk.telegram.router import TelegramConversationMessage, TelegramDecision, build_telegram_router
from memetalk.telegram.runtime import DirectTelegramSearchClient

logger = logging.getLogger(__name__)

TELEGRAM_HISTORY_KEY = "conversation_history"
TELEGRAM_HISTORY_LIMIT = 8


def validate_telegram_settings(settings: AppSettings) -> str:
    if not settings.telegram_enabled:
        raise RuntimeError("Telegram chat is disabled. Enable it in Settings or set MEMETALK_TELEGRAM_ENABLED=1.")
    token = (settings.telegram_bot_token or "").strip()
    if not token:
        raise RuntimeError(
            "Telegram bot token is not configured. Set it in Settings or provide MEMETALK_TELEGRAM_BOT_TOKEN."
        )
    return token


def create_application(
    settings: AppSettings | None = None,
    container: AppContainer | None = None,
):
    active_settings = settings or AppSettings.from_env()
    telegram_token = validate_telegram_settings(active_settings)
    active_container = container or build_container(active_settings)
    router = build_telegram_router(active_settings)
    search_client = DirectTelegramSearchClient(active_container, active_settings.search_candidate_k_default)

    try:
        from telegram.constants import ChatAction
        from telegram.ext import Application, CommandHandler, MessageHandler, filters
    except ImportError as exc:
        raise RuntimeError("Telegram support is not installed. Install with `pip install -e .[telegram]`.") from exc

    async def respond_to_decision(message, decision: TelegramDecision) -> str:
        if decision.action == "text":
            text_response = decision.text_response or "我先用文字回你。"
            await message.reply_text(text_response)
            return text_response

        meme_sent, history_summary = await _send_meme_only_reply(message, decision, search_client)
        if meme_sent:
            return history_summary or _summarize_meme_reply(decision)

        fallback = decision.text_response or "找不到適合的梗圖，但我有在認真聽你說話。"
        await message.reply_text(fallback)
        return fallback

    async def start_command(update, context) -> None:
        if update.message is None:
            return
        _reset_conversation_history(context.chat_data)
        await update.message.reply_text(
            "嗨！我是 MemeTalk 的 Telegram 梗圖機器人。\n"
            "直接丟一句話給我，我會判斷該用文字、梗圖，或兩者一起回你。\n"
            "輸入 /help 查看使用方式。"
        )

    async def help_command(update, context) -> None:
        if update.message is None:
            return
        await update.message.reply_text(
            "使用方式：直接傳送文字訊息。\n\n"
            "我會用目前 MemeTalk 的 provider 與搜尋設定，決定要回文字、梗圖，或文字加梗圖。\n"
            "如果想關閉 Telegram，回到 MemeTalk Settings 把 Telegram 開關關掉即可。"
        )

    async def handle_message(update, context) -> None:
        if update.message is None or not update.message.text:
            return

        decision: TelegramDecision
        user_text = update.message.text.strip()
        await update.message.chat.send_action(ChatAction.TYPING)
        conversation_history = _load_conversation_history(context.chat_data)

        try:
            decision = await context.bot_data["router"].decide(
                user_text,
                conversation_history=conversation_history,
            )
        except Exception:
            logger.error("Telegram routing failed", exc_info=True)
            _append_conversation_history(context.chat_data, "user", user_text)
            _append_conversation_history(context.chat_data, "assistant", "抱歉，我暫時無法處理你的訊息。")
            await update.message.reply_text("抱歉，我暫時無法處理你的訊息。")
            return

        logger.info("Telegram decision: action=%s query=%s", decision.action, decision.search_query)

        assistant_summary = await respond_to_decision(update.message, decision)
        _append_conversation_history(context.chat_data, "user", user_text)
        _append_conversation_history(context.chat_data, "assistant", assistant_summary)

    async def post_shutdown(application) -> None:
        await application.bot_data["search_client"].close()

    application = (
        Application.builder()
        .token(telegram_token)
        .post_shutdown(post_shutdown)
        .build()
    )
    application.bot_data["router"] = router
    application.bot_data["search_client"] = search_client

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return application


def run_polling(settings: AppSettings | None = None) -> None:
    active_settings = settings or AppSettings.from_env()
    application = create_application(active_settings)
    logger.info("Starting Telegram bot with long polling using provider=%s", active_settings.provider_backend)
    application.run_polling()


async def _send_meme_only_reply(
    message,
    decision: TelegramDecision,
    search_client: DirectTelegramSearchClient,
) -> tuple[bool, str | None]:
    if not decision.search_query:
        return False, None
    try:
        results = await search_client.search_memes(
            query=decision.search_query,
            mode=decision.search_mode,
            top_n=3,
        )
        if not results:
            return False, None
        top = results[0]
        try:
            image_bytes = await search_client.get_meme_image(top.image_id)
            await message.reply_photo(
                photo=io.BytesIO(image_bytes),
                caption=None,
            )
            return True, _summarize_meme_reply(decision)
        except Exception:
            logger.warning("Telegram meme image read failed", exc_info=True)
            if top.reason:
                fallback_text = f"（找到梗圖但無法載入）{top.reason}"
                await message.reply_text(fallback_text)
                return True, fallback_text
            return False, None
    except Exception:
        logger.warning("Telegram meme search failed", exc_info=True)
        return False, None


def _summarize_meme_reply(decision: TelegramDecision) -> str:
    if decision.search_query:
        return f"已用梗圖回覆（搜尋詞：{decision.search_query}）"
    return "已用梗圖回覆"


def _load_conversation_history(chat_data: MutableMapping[str, Any]) -> list[TelegramConversationMessage]:
    raw_history = chat_data.get(TELEGRAM_HISTORY_KEY, [])
    history: list[TelegramConversationMessage] = []
    for item in raw_history:
        try:
            if isinstance(item, TelegramConversationMessage):
                history.append(item)
            else:
                history.append(TelegramConversationMessage.model_validate(item))
        except Exception:
            continue
    if len(history) > TELEGRAM_HISTORY_LIMIT:
        history = history[-TELEGRAM_HISTORY_LIMIT:]
    chat_data[TELEGRAM_HISTORY_KEY] = history
    return history


def _append_conversation_history(
    chat_data: MutableMapping[str, Any],
    role: str,
    content: str,
) -> None:
    try:
        entry = TelegramConversationMessage(role=role, content=content)
    except Exception:
        return
    history = _load_conversation_history(chat_data)
    history.append(entry)
    if len(history) > TELEGRAM_HISTORY_LIMIT:
        del history[:-TELEGRAM_HISTORY_LIMIT]
    chat_data[TELEGRAM_HISTORY_KEY] = history


def _reset_conversation_history(chat_data: MutableMapping[str, Any]) -> None:
    chat_data.pop(TELEGRAM_HISTORY_KEY, None)
