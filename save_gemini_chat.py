import asyncio
import argparse
from pathlib import Path
from playwright.async_api import async_playwright


async def save_chat(url: str, output: str, headless: bool):
    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            user_data_dir="./chrome_profile",
            headless=headless
        )
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(5000)

        messages = []

        # Идём по блокам каждого хода чата
        turns = await page.query_selector_all(".share-turn-viewer")

        if turns:
            for turn in turns:
                # Вопрос пользователя
                user_el = await turn.query_selector(".query-text")
                if user_el:
                    text = (await user_el.inner_text()).strip()
                    if text:
                        messages.append(f"## 🧑 Вы\n\n{text}\n")

                # Ответ Gemini
                model_el = await turn.query_selector(".message-content")
                if model_el:
                    text = (await model_el.inner_text()).strip()
                    if text:
                        messages.append(f"## 🤖 Gemini\n\n{text}\n")
        else:
            # Fallback: берём отдельно все вопросы и ответы
            user_turns = await page.query_selector_all(".query-text")
            model_turns = await page.query_selector_all(".message-content")
            for i in range(max(len(user_turns), len(model_turns))):
                if i < len(user_turns):
                    text = (await user_turns[i].inner_text()).strip()
                    if text:
                        messages.append(f"## 🧑 Вы\n\n{text}\n")
                if i < len(model_turns):
                    text = (await model_turns[i].inner_text()).strip()
                    if text:
                        messages.append(f"## 🤖 Gemini\n\n{text}\n")

        if not messages:
            print("⚠️  Сообщения не найдены.")
        else:
            title_el = await page.query_selector("h1")
            title = (await title_el.inner_text()).strip() if title_el else "Gemini Chat"
            header = f"# {title}\n\nURL: {url}\n\n---\n\n"
            Path(output).write_text(
                header + "\n---\n\n".join(messages),
                encoding="utf-8"
            )
            print(f"✅ Сохранено {len(messages)} сообщений → {output}")

        await browser.close()


def main():
    parser = argparse.ArgumentParser(
        description="Сохраняет публичный чат Gemini (/share/...) в Markdown-файл"
    )
    parser.add_argument("url", help="https://gemini.google.com/share/...")
    parser.add_argument("-o", "--output", default="gemini_chat.md")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    asyncio.run(save_chat(args.url, args.output, args.headless))


if __name__ == "__main__":
    main()