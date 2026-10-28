"""AutoDL 网页自动化（Playwright）：官方 API 创建 Pro 实例返回「无当前资源访问权限」时的替代方案。

浏览器内核在 I:/cache/ms-playwright，登录状态保存在 I:/cache/autodl-browser（持久化 profile），
登录只需人工做一次（扫码 / 短信），脚本不保存账号密码。

用法：
  python -m scripts.autodl_browser login     # 打开浏览器窗口，人工登录，登录成功后自动关闭
  python -m scripts.autodl_browser inspect   # 保存控制台和租用页面的截图与 HTML，用来编写选择器
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", r"I:\cache\ms-playwright")

PROFILE = r"I:\cache\autodl-browser"
CONSOLE = "https://www.autodl.com/console/homepage/personal"
MARKET = "https://www.autodl.com/market/list"
OUT = Path("outputs/autodl/pages")


def open_context(p, headless: bool):
    return p.chromium.launch_persistent_context(PROFILE, headless=headless, viewport={"width": 1440, "height": 900},
                                                locale="zh-CN")


def logged_in(page) -> bool:
    page.goto(CONSOLE, wait_until="domcontentloaded")
    page.wait_for_timeout(3000)
    return "login" not in page.url


def fill_login(page) -> None:
    """有 AUTODL_PHONE / AUTODL_PASSWORD 环境变量时代填账号密码（不落盘）；AUTODL_AREA 为区号，如 81。"""
    phone, pwd = os.environ.get("AUTODL_PHONE"), os.environ.get("AUTODL_PASSWORD")
    if not (phone and pwd):
        return
    area = os.environ.get("AUTODL_AREA", "86")
    page.wait_for_selector("input[placeholder='请输入手机号']", timeout=20000)
    if area != "86":
        page.locator("input[placeholder='请选择']").first.click()
        page.wait_for_timeout(800)
        page.locator(".el-select-dropdown__item:visible", has_text=f"+{area}").first.click()
    page.fill("input[placeholder='请输入手机号']", phone)
    page.fill("input[placeholder='请输入密码']", pwd)
    page.locator("button.el-button--primary", has_text="登录").first.click()
    print("已填写账号密码并点击登录；如出现验证码请在窗口中完成", flush=True)


def cmd_login(timeout_min: int) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        ctx = open_context(p, headless=False)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(CONSOLE)
        print("请在打开的浏览器窗口里登录 AutoDL ...", flush=True)
        fill_login(page)
        t0 = time.time()
        while time.time() - t0 < timeout_min * 60:
            try:
                page.wait_for_timeout(3000)
            except Exception:  # noqa: BLE001 窗口被人工关掉
                print("浏览器窗口已关闭", flush=True)
                return
            if "login" not in page.url and "autodl.com/console" in page.url:
                print(f"已登录：{page.url}", flush=True)
                page.wait_for_timeout(3000)
                break
        else:
            print("超时未登录", flush=True)
        ctx.close()


def cmd_inspect() -> None:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        ctx = open_context(p, headless=True)
        page = ctx.new_page()
        if not logged_in(page):
            print("未登录，先运行 login", flush=True)
            ctx.close()
            return
        for name, url in (("console", CONSOLE), ("market", MARKET)):
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(3000)
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
            (OUT / f"{name}.html").write_text(page.content(), encoding="utf-8")
            print(f"saved {name}: {page.url}", flush=True)
        ctx.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["login", "inspect"])
    ap.add_argument("--timeout_min", type=int, default=15)
    args = ap.parse_args()
    if args.cmd == "login":
        cmd_login(args.timeout_min)
    else:
        cmd_inspect()


if __name__ == "__main__":
    main()
