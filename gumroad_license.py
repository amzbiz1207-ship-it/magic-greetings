"""Verify Magic Greetings licenses on the server."""

import re

import requests


class LicenseUnavailable(Exception):
    """Gumroad could not complete verification."""


def verify_license(product_id: str, license_key: str) -> str | None:
    """Return the verified sale ID or None for a rejected key."""
    if not product_id:
        raise ValueError("The Gumroad product ID is required.")
    key = license_key.strip().upper()
    if not re.fullmatch(r"[A-F0-9]{8}(?:-[A-F0-9]{8}){3}", key):
        return None

    try:
        response = requests.post(
            "https://api.gumroad.com/v2/licenses/verify",
            data={"product_id": product_id, "license_key": key,
                  "increment_uses_count": "false"},
            timeout=(4, 10),
            allow_redirects=False,
        )
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise LicenseUnavailable("Gumroad verification is temporarily unavailable.")
        payload = response.json()
    except (requests.RequestException, ValueError):
        raise LicenseUnavailable("Gumroad verification is temporarily unavailable.") from None

    if not isinstance(payload, dict):
        raise LicenseUnavailable("Unexpected Gumroad response.")
    if payload.get("success") is False:
        return None
    purchase = payload.get("purchase")
    if payload.get("success") is not True or not isinstance(purchase, dict):
        raise LicenseUnavailable("Unexpected Gumroad response.")
    if purchase.get("product_id") != product_id:
        return None
    if purchase.get("refunded") or purchase.get("chargebacked"):
        return None
    if purchase.get("disputed") and purchase.get("dispute_won") is not True:
        return None
    sale_id = purchase.get("sale_id")
    if not isinstance(sale_id, str) or not sale_id:
        raise LicenseUnavailable("Missing Gumroad sale ID.")
    return sale_id
