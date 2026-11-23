# Magic Greetings — launch checkpoint, 23 November 2026

## Current publication
- Public Streamlit app: https://magic-greetings-h2ufnyxamtyr32a7rhuurw.streamlit.app/
- Repository: amzbiz1207-ship-it/magic-greetings; branch: scene-preview.
- App release: f8d0335ef1d11dcd1d6ece7da036620e7430ff17.
- No new domain or hosting purchase is needed for this preview.
- Public landing and catalogue opened successfully in the browser on this date.
- Existing preview uses a shared trial password; it is NOT paid customer access.

## Completed checks
- 15 request/generation unit tests passed.
- Controlled AppTest flow passed: create, pending state, rerun without a duplicate request, completion, greeting composition and PNG download.
- Controlled refinement flow passed.
- HTTP 401, 429 and 503 show readable errors and release pending resume state.
- Full landing to scene selection to uploader passed.
- Source short-side checks: 1000 pixels accepted; 999, 504 and 300 pixels rejected.
- These are local deterministic tests, not new paid API generations.
- Earlier real generation worked; the owner also confirmed a successful Royal card.
- One refinement is currently limited by the session's version list, not by a persistent purchased entitlement.

## Customer guidance
- Photo requirements are displayed before upload.
- Check original file, visible faces, lighting, minimum resolution and face size.
- An automatic pass does not guarantee likeness and does not validate every advisory rule.
- A new RU/EN help section explains Streamlit wake-up, generation status, errors and downloading before closing the session.
- Wake-up instructions must also accompany the sales link: a sleeping Streamlit page appears before the app itself can display help.

## Remaining before accepting money
1. Select the Stripe account/mode. The only connector account currently exposed is Mila Taylor sandbox (test mode). User selection is pending; no account-specific integration planning or writes have been performed.
2. Agree the exact bundle and price. About £1 per card and one correction per card are the owner's direction; 7 cards for £7 is a proposal, not an approved published tariff.
3. Obtain the Stripe integration plan before writing payment integration code.
4. Provide durable order/credit storage. Streamlit session_state alone is unsuitable; do not use its ephemeral filesystem as the sole ledger.
5. Create hosted checkout and grant access only after server-side verification of a paid payment for the configured product, currency and amount.
6. Use individual access and atomic credit reservation, not a shared customer password.
7. Test duplicate callback/webhook delivery, simultaneous generation, refresh, failed payment, failed generation, credit release and exactly one correction per purchased card.
8. Confirm support contact and customer-facing purchase/refund/privacy information before publication.
9. Run test checkout end to end, then configure live credentials securely and perform an owner-approved live check.

## Cost measurement
Actual API usage logging is implemented. No actual per-card cost was measured in this checkpoint. Do not present illustrative token estimates as an observed cost or a maximum.

## Release boundary
The app is publicly accessible as a preview. Commercial payment, durable entitlement and automated post-payment access are NOT ready or advertised as ready.
