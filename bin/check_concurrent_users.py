import argparse
import asyncio
import json
import re
import shutil
import time

from playwright.async_api import async_playwright, expect


async def wait_complete(page):
    errors = page.locator('[data-testid="stException"]')
    await (
        page.get_by_label("Find topics", exact=True)
        .or_(errors)
        .first.wait_for(timeout=240_000)
    )
    await page.locator('[data-testid="stStatusWidgetRunningIcon"]').wait_for(
        state="hidden", timeout=240_000
    )
    assert await errors.count() == 0, await errors.all_text_contents()


async def main(url, chunks, device):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=shutil.which("chromium"),
            args=["--no-sandbox"],
        )
        try:
            contexts = [await browser.new_context() for _ in range(8)]
            pages = [await context.new_page() for context in contexts]

            async def prepare(index, page):
                await page.goto(url)
                await page.get_by_role("button", name="Compute!", exact=True).wait_for(
                    timeout=120_000
                )
                await page.get_by_text(device, exact=True).click()
                await page.get_by_role(
                    "spinbutton", name="Chunks per document (0 = all)", exact=True
                ).fill(str(chunks if index < 3 else chunks + 5))

            await asyncio.gather(*(prepare(i, page) for i, page in enumerate(pages)))
            start = time.monotonic()
            await asyncio.gather(
                *(
                    page.get_by_role("button", name="Compute!", exact=True).click()
                    for page in pages
                )
            )
            await asyncio.gather(*(wait_complete(page) for page in pages))
            print(
                json.dumps(
                    {
                        "phase": "eight computations",
                        "seconds": round(time.monotonic() - start, 2),
                        "metrics": [
                            await page.locator(
                                '[data-testid="stMetricValue"]'
                            ).all_text_contents()
                            for page in pages
                        ],
                    }
                ),
                flush=True,
            )

            async def interact(index, page):
                if index < 3:
                    query = page.get_by_label("Find topics", exact=True)
                    await query.fill(["学校", "動物", "戦争"][index])
                    await query.press("Enter")
                    await page.get_by_text("Topic Query", exact=True).wait_for(
                        state="attached", timeout=240_000
                    )
                else:
                    model_info = page.get_by_text(re.compile(r"^Model cache/"))
                    previous = await model_info.inner_text()
                    await page.get_by_role(
                        "spinbutton", name="Chunk size (tokens)", exact=True
                    ).fill("150" if index == 3 else "200")
                    await page.get_by_role(
                        "button", name="Compute!", exact=True
                    ).click()
                    await expect(model_info).not_to_have_text(previous, timeout=240_000)
                await wait_complete(page)

            start = time.monotonic()
            await asyncio.gather(*(interact(i, page) for i, page in enumerate(pages)))
            print(
                json.dumps(
                    {
                        "phase": "three searches and five recomputations",
                        "seconds": round(time.monotonic() - start, 2),
                        "exceptions": [
                            await page.locator(
                                '[data-testid="stException"]'
                            ).all_text_contents()
                            for page in pages
                        ],
                    }
                ),
                flush=True,
            )
        finally:
            await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check eight independent sessions against a running app using the selected device. Run with uv tool run --from playwright python bin/check_concurrent_users.py URL."
    )
    parser.add_argument("url")
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.chunks, args.device))
