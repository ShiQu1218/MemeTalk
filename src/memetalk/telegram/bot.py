from __future__ import annotations

import io
import logging

from memetalk.app.container import AppContainer, build_container
from memetalk.config import AppSettings
from memetalk.telegram.router import TelegramDecision, build_telegram_router
from memetalk.telegram.runtime import DirectTelegramSearchClient

logger = logging.getLogger(__name__)


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

    async def start_command(update, context) -> None:
        if update.message is None:
            return
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

        try:
            decision = await context.bot_data["router"].decide(user_text)
        except Exception:
            logger.error("Telegram routing failed", exc_info=True)
            await update.message.reply_text("抱歉，我暫時無法處理你的訊息。")
            return

        logger.info("Telegram decision: action=%s query=%s", decision.action, decision.search_query)

        if decision.action == "text":
            await update.message.reply_text(decision.text_response or "我先用文字回你。")
            return

        meme_sent = False
        if decision.search_query:
            try:
                results = await context.bot_data["search_client"].search_memes(
                    query=decision.search_query,
                    mode=decision.search_mode,
                    top_n=3,
                )
                if results:
                    top = results[0]
                    try:
                        image_bytes = await context.bot_data["search_client"].get_meme_image(top.image_id)
                        await update.message.reply_photo(
                            photo=io.BytesIO(image_bytes),
                            caption=top.reason or None,
                        )
                        meme_sent = True
                    except Exception:
                        logger.warning("Telegram meme image read failed", exc_info=True)
                        if top.reason:
                            await update.message.reply_text(f"（找到梗圖但無法載入）{top.reason}")
                            meme_sent = True
            except Exception:
                logger.warning("Telegram meme search failed", exc_info=True)

        if decision.action == "both" and decision.text_response:
            await update.message.reply_text(decision.text_response)
        elif decision.action == "meme" and not meme_sent:
            fallback = decision.text_response or "找不到適合的梗圖，但我有在認真聽你說話。"
            await update.message.reply_text(fallback)

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
