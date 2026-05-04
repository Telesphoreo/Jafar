#!/usr/bin/env python3
"""
Helper script to add a Twitter account to twscrape using exported cookies.

Automatically assigns a proxy from config.yaml in round-robin fashion.

Usage:
    python add_account.py <username> <cookies.json>

Example:
    python add_account.py myusername cookies.json
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import yaml
from twscrape import API


def load_proxies() -> list[str]:
    """Load proxies from config.yaml or .env."""
    config_path = Path(__file__).parent / "config.yaml"

    # Try config.yaml first
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        proxies = config.get("twitter", {}).get("proxies", [])
        if proxies:
            return proxies

    # Fall back to .env TWITTER_PROXIES
    env_proxies = os.getenv("TWITTER_PROXIES", "")
    if env_proxies:
        return [p.strip() for p in env_proxies.split(",") if p.strip()]

    return []


def parse_cookies_file(filepath: str) -> str:
    """
    Parse cookies from various JSON formats and return as a cookie string.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    cookies = {}

    if isinstance(data, list):
        # Format: [{"name": "auth_token", "value": "xxx"}, ...]
        for cookie in data:
            name = cookie.get("name") or cookie.get("Name")
            value = cookie.get("value") or cookie.get("Value")
            if name and value:
                cookies[name] = value
    elif isinstance(data, dict):
        if "cookies" in data:
            for cookie in data["cookies"]:
                name = cookie.get("name") or cookie.get("Name")
                value = cookie.get("value") or cookie.get("Value")
                if name and value:
                    cookies[name] = value
        else:
            cookies = data

    cookie_string = "; ".join(f"{k}={v}" for k, v in cookies.items())
    return cookie_string


async def add_account_with_cookies(
    username: str,
    cookies_file: str,
    db_path: str = "accounts.db",
    proxy: str | None = None,
):
    """Add a Twitter account using cookies from a JSON file."""

    print(f"Reading cookies from: {cookies_file}")
    cookie_string = parse_cookies_file(cookies_file)

    cookie_names = [c.split("=")[0] for c in cookie_string.split("; ")]
    print(f"Found {len(cookie_names)} cookies")

    required = ["auth_token", "ct0"]
    missing = [r for r in required if r not in cookie_names]
    if missing:
        print(f"ERROR: Missing required cookies: {missing}")
        print("Make sure you're logged into Twitter when exporting cookies.")
        sys.exit(1)

    print(f"Required cookies found: {required}")

    if proxy:
        print(f"Proxy: {proxy[:40]}...")

    api = API(db_path)

    print(f"\nRemoving any existing accounts...")
    try:
        await api.pool.delete_accounts(["twitter_user", username])
    except Exception:
        pass  # Ignore if accounts don't exist

    print(f"Adding account '{username}' with cookies...")

    await api.pool.add_account(
        username=username,
        password="cookie_based_auth",
        email=f"{username}@cookie.local",
        email_password="",
        cookies=cookie_string,
        proxy=proxy,
    )

    print("\nAccount added! Checking status...")

    accounts = await api.pool.accounts_info()
    for acc in accounts:
        if acc["username"] == username:
            print(f"\n  Username: {acc['username']}")
            print(f"  Active: {acc['active']}")
            if acc.get("proxy"):
                print(f"  Proxy: {acc['proxy'][:40]}...")

    print(f"\nRun 'twscrape accounts' to verify.")


async def main():
    if len(sys.argv) < 3:
        print("Usage: python add_account.py <twitter_username> <cookies.json>")
        print("\nExample:")
        print("  python add_account.py myusername cookies.json")
        sys.exit(1)

    username = sys.argv[1]
    cookies_file = sys.argv[2]

    if not Path(cookies_file).exists():
        print(f"Error: File not found: {cookies_file}")
        sys.exit(1)

    proxies = load_proxies()
    proxy = None

    if proxies:
        api = API("accounts.db")
        try:
            accounts = await api.pool.accounts_info()
            existing_usernames = [a["username"] for a in accounts]

            # If this is a new account, assign next proxy in rotation
            # If re-adding existing account, find its original position
            if username in existing_usernames:
                account_position = sorted(existing_usernames).index(username)
            else:
                account_position = len(accounts)
        except Exception:
            account_position = 0

        # Assign proxy round-robin based on account position
        proxy_index = account_position % len(proxies)
        proxy = proxies[proxy_index]
        print(f"Assigning proxy {proxy_index + 1}/{len(proxies)} to this account (position {account_position})")
    else:
        print("No proxies configured in config.yaml")

    await add_account_with_cookies(username, cookies_file, proxy=proxy)


if __name__ == "__main__":
    asyncio.run(main())
