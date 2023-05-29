from playwright.sync_api import Playwright, sync_playwright, expect
from random import randrange


def check_params(page):
    chunksize = page.get_by_role("spinbutton", name="Maximum chunksize").input_value()
    chunks = page.get_by_role(
        "spinbutton", name="Number of chunks per doc (0 for all)"
    ).input_value()
    print(f"\n{chunksize} {chunks}")
    page.get_by_text(f"Chunksize: {chunksize}").click()
    page.get_by_text(f"Chunks/doc: {chunks}").click()
    expect(page.get_by_text("Running...")).to_have_count(0, timeout=0)


def r():
    return randrange(1, 20)


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.set_default_timeout(0)
    page.goto("http://localhost:8501/")
    expect(page.locator(".stException")).to_have_count(0)
    expect(page.get_by_text("Running...")).to_have_count(0, timeout=0)
    page.get_by_role("button", name="Debug information").click()
    check_params(page)

    # Intertopic distance map
    page.locator(".slider-rail-touch-rect").click(timeout=150000)
    expect(page.locator(".stException")).to_have_count(0)

    # 文章の可視化
    # page.locator("div").filter(has_text="岡本綺堂").first.click()
    # page.get_by_role("option", name="江戸川乱歩").get_by_text("江戸川乱歩").click()
    # page.locator("div").filter(has_text="宇宙怪人").first.click()
    # page.get_by_text("少年探偵団").click()

    # expect(page.locator(".stException")).to_have_count(0)

    # page.get_by_test_id("stHorizontalBlock").get_by_role("button").nth(1).click()
    # expect(page.locator(".stException")).to_have_count(0)
    # page.locator(".x11y11 > .ndrag").click(timeout=0)
    # expect(page.locator(".stException")).to_have_count(0)
    # page.get_by_test_id("stHorizontalBlock").get_by_role("button").nth(3).click()
    # expect(page.locator(".stException")).to_have_count(0)
    # page.get_by_test_id("stAppViewContainer").locator("div").filter(
    #     has_text="BERTopicを使用したトピックモデル"
    # ).first.click(timeout=60000)
    # expect(page.locator(".stException")).to_have_count(0)

    # page.get_by_role("textbox", name="文章の可視化").click()
    # page.get_by_role("textbox", name="文章の可視化").fill("どのようなトピックがトピックになるかなあ？？Test")
    # page.get_by_role("textbox", name="文章の可視化").press("Control+Enter")
    # expect(page.locator(".stException")).to_have_count(0)

    # Settings

    # page.get_by_test_id("stForm").get_by_title("open").nth(1).click()
    # page.get_by_text("MaximalMarginalRelevance").click()
    # page.get_by_role(
    #     "combobox", name="Selected MaximalMarginalRelevance. Topic representation"
    # ).press("Enter")

    # page.get_by_test_id("stForm").get_by_title("open").nth(2).click()
    # page.get_by_role("option", name="Sudachi").click()

    # chunksize/chunks
    page.get_by_role("spinbutton", name="Maximum chunksize").click()
    for _ in range(r()):
        page.locator(".step-up").first.click()

    page.get_by_role("spinbutton", name="Number of chunks per doc (0 for all)").click()
    for _ in range(r()):
        page.locator(".step-up").first.click()

    check_params(page)
    expect(page.locator(".stException")).to_have_count(0)

    # Reduce topics
    # page.get_by_test_id("stSidebar").locator("span").click()

    # expect(page.locator(".stException")).to_have_count(0)
    # check_params(page)

    # expect(page.get_by_text("Running...")).to_have_count(0, timeout=0)
    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
